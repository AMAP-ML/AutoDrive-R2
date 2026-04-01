# Copyright 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
accelerate launch \
    --config_file=deepspeed_zero2.yaml \
    train_video_llm.py \
    --dataset_name mfarre/simplevideoshorts \
    --model_name_or_path Qwen/Qwen2-VL-7B-Instruct \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir video-llm-output \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
"""

import os
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

from typing import List, Dict, Any

def get_current_device():
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"

def download_video(url: str, folder: str = '/tmp/videos/') -> str:
    """Download video if not already present locally."""
    filename = url.split("/")[-1]
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return local_path
    except requests.RequestException as e:
        raise Exception(f"Failed to download video: {e}")

def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    

    system_message = "You are a helpful assistant"
    
    
    # QUESTION_TEMPLATE = (
    #     "{Question}\n"
    #     "Please think about this question as if you were a human pondering deeply. "
    #     "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    #     "It's encouraged to include self-reflection or verification in the reasoning process. "
    #     "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    # )
    
    QUESTION_TEMPLATE = (
    """
    I will provide you with an image, a driving-related question. 
    Please think about this question as if you were a human pondering deeply.
    Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions
    It's encouraged to include self-reflection or verification in the reasoning process. 
    Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.

    ### **Input Format:**  
    - **System Instructions**: {original_task}  
    - **Past Vehicle Status**: {original_information}  
    - **Prediction Task**: {original_problem}  

    ### **Output Format:**  

    ### **1. Visual Analysis**  
    "Image analysis results:  
    - Vehicle's intended direction: Left turn (steering wheel angle: θ rad)  
    - Obstacles ahead:  
    * Car detected ahead (moving right/left/straight)  
    * Pedestrian crossing road (left/right side)  
    - Traffic signal: signal_status detected (red / green / yellow)"  

    ### **2. Motion Modeling**  
    "Using historical data in **Past Vehicle Status** with N time points:  
    t₁: [x=x₁, y=y₁], v=v₁m/s, a=(a_x₁, a_y₁)m/s²  
    t₂: [x=x₂, y=y₂], v=v₂m/s, a=(a_x₂, a_y₂)m/s²  
    ...  
    tₙ: [x=xₙ, y=yₙ], v=vₙm/s, a=(a_xₙ, a_yₙ)m/s²  

    Calculations:  
    - Average acceleration:  
    a_avg_x = (Σa_x_i)/N ≈ a_x_avgm/s²  
    a_avg_y = (Σa_y_i)/N ≈ a_y_avgm/s²  
    - Velocity prediction:  
    v_x = vₙ + a_avg_x × Δt ≈ v_t0 + a_x_avg × Δt  
    v_y = vₙ + a_avg_y × Δt ≈ v_t0 + a_y_avg × Δt  
    - Position prediction:  
    x(t+1) = xₙ + v_x × Δt + 0.5 × a_avg_x × Δt² ≈ x_t0 + v_x × Δt + 0.5 × a_x_avg × Δt²  
    y(t+1) = yₙ + v_y × Δt + 0.5 × a_avg_y × Δt² ≈ y_t0 + v_y × Δt + 0.5 × a_y_avg × Δt²  
    - Lateral offset: Δy = v × tan(θ) = v_t0 × tan(θ)"  

    ### **3. Logical Deductions**  
    "Safety check:  
    - If following this trajectory, will the vehicle:  
    * Run a red light? → yes/no  
    * Collide with car ahead? → yes/no  
    * Hit pedestrian crossing? → yes/no  
    - Conclusion: recommended_action (e.g., 'Stop immediately', 'Reduce speed to 5m/s')"  

    ### **4. Self-Reflection Validation**  
    "Validation:  
    - Assumption check:  
    * Predicted position (x=x_pred, y=y_pred) requires average speed of v_requiredm/s  
    * Is this speed achievable with vehicle's acceleration history? → yes/no  
    - Adjustment:  
    * If not feasible → Modify trajectory by reducing speed or increasing stopping distance"  
    </think>
    <answer>(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)</answer>
    """
    )

    information = example['information']
    problem = example['problem']
    task = example['task']

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        {
            "role": "user",
            "content": [
                {

                    "type": example['data_type'],
                    # example['data_type']: os.path.join("/mnt/xmap_nas_alg/yzl/Auto_Drive/AFile/datasets", example['path'])
                    example['data_type']: os.path.join("/mnt/xmap_nas_alg/yzl/Auto_Drive/AFile/datasets", example['path'])
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(original_information=information, original_problem=problem, original_task=task)
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['process'] + "\n" + example['solution']}]
        }
    ]
    
    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # # Quantization configuration for 4-bit training
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # Prepare dataset
    prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
