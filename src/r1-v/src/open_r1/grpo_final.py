# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

all_rewards = []

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )


def accuracy_reward(completions, solution, **kwargs):

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m, n = len(ref_words), len(hyp_words)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): d[i][0] = i
        for j in range(n + 1): d[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                d[i][j] = (
                    d[i - 1][j - 1] if ref_words[i - 1] == hyp_words[j - 1]
                    else 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
                )
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure

    def parse_coordinates(s):
        numbers_str = re.findall(r'-?\d+\.?\d*', s)
        try:
            numbers = [float(num) for num in numbers_str]
        except ValueError:
            return None
        if len(numbers) % 2 != 0:
            return None
        coords = [(numbers[i], numbers[i + 1]) for i in range(0, len(numbers), 2)]
        return coords

    def parse_driving_data(s):
        # 提取坐标
        coords = parse_coordinates(s)
        # 提取转向角 (steering angle)
        steering_match = re.search(r'steering:\s*([-]?\d+\.?\d*\s*,?\s*)+', s)
        steering = [float(x.strip()) for x in steering_match.group(1).split(',')] if steering_match else []
        # 提取速度 (velocity)
        velocity_match = re.search(r'velocity:\s*([-]?\d+\.?\d*\s*,?\s*)+', s)
        velocity = [float(x.strip()) for x in velocity_match.group(1).split(',')] if velocity_match else []
        return coords, steering, velocity

    question_type = kwargs['problem_type'][0]

    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    logs = []  # 收集所有样本的日志
    count = 0

    for content, sol in zip(contents, solution):
        try:
            count += 1
            output_ans = extract_answer(content)
            gt_ans = extract_answer(sol)

            if question_type == "multiple choice":
                reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "driving":
                # 解析预测和真实数据
                pred_coords, pred_steering, pred_velocity = parse_driving_data(output_ans)
                true_coords, true_steering, true_velocity = parse_driving_data(gt_ans)

                if not pred_coords or not true_coords:
                    pred_coords = "None"
                    true_coords = "None"
                    reward = 0.0
                elif len(pred_coords) != len(true_coords):
                    reward = 0.0
                else:
                    # 计算位置误差
                    pos_error = sum((x_pred - x_gt)**2 + (y_pred - y_gt)**2 for (x_pred, y_pred), (x_gt, y_gt) in zip(pred_coords, true_coords)) / len(pred_coords)
                    
                    # 计算方向误差
                    steer_error = sum((s_pred - s_gt)**2 for s_pred, s_gt in zip(pred_steering, true_steering)) / len(pred_steering) if pred_steering and true_steering else 0.0
                    
                    # 计算速度误差
                    vel_error = sum((v_pred - v_gt)**2 for v_pred, v_gt in zip(pred_velocity, true_velocity)) / len(pred_velocity) if pred_velocity and true_velocity else 0.0

                    # Temporal smoothness
                    temporal_smoothness = 0.0
                    if len(pred_velocity) > 1:
                        for i in range(1, len(pred_velocity)):
                            temporal_smoothness += (pred_velocity[i] - pred_velocity[i-1])**2
                        temporal_smoothness /= (len(pred_velocity) - 1)
                    else:
                        temporal_smoothness = 0.0  # Default if only one velocity point

                    
                    # 加权总误差
                    w_pos = 1
                    w_steer = 0
                    w_vel = 1
                    w_temporal = 1
                    
                    total_error = (
                        pos_error * w_pos +
                        steer_error * w_steer +
                        vel_error * w_vel +
                        temporal_smoothness * w_temporal
                    )
                    
                    # 计算奖励
                    reward = 1.0 / (1.0 + total_error)

                print("Content", content)
                print("Solution", sol)
                print("pred_coords", pred_coords)
                print("true_coords", true_coords)     
                print("reward: ", reward)    

                # 将日志信息添加到 logs
                logs.append(f"------------- {current_time} Driving Sample {count} -------------\n")
                logs.append(f"Content: {content}\n")
                logs.append(f"Solution: {sol}\n")
                logs.append(f"pred_coords: {pred_coords}\n")
                logs.append(f"true_coords: {true_coords}\n")
                logs.append(f"Reward: {reward}\n")
            else:
                reward = 0.0

        except Exception as e:
            print(f"Error in reward_fn for question_type '{question_type}': {e}")
            reward = 0.0

        rewards.append(reward)

    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    
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

        
    def make_conversation_image_and_video(example):

        information = example['information']
        problem = example['problem']
        task = example['task']

        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(original_information=information, original_problem=problem, original_task=task)
                        }
                        ]
                }]
            }
        
        return msg

    
    dataset = dataset.map(make_conversation_image_and_video)

    from transformers import TrainerCallback, TrainerState, TrainerControl

    # Initialize the GRPO trainer
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)



if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)