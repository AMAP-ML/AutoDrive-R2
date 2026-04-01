import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch

from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse


BSZ = 64


parser = argparse.ArgumentParser(description="Evaluation benchmark")
args = parser.parse_args()

PROMPT_PATH = f"/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AData/nuscenes_test.json"

MODEL_PATH = "/data/oss_bucket_0/yzl/Auto_Drive/AFile/models/grpo_video_7B_6k/checkpoint-750"

OUTPUT_PATH = f"/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AScripts/output/nuscene/grpo_video_7B_6k.json"

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    max_model_len = 8192 * 2,
    gpu_memory_utilization=0.9,
)


sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.001,
    max_tokens=4096,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

if PROMPT_PATH.endswith('.jsonl'):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
elif PROMPT_PATH.endswith('.json'):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    raise ValueError("Input file must be .json or .jsonl")

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

messages = []
for x in data:
    information = x['information']
    problem = x['problem']
    task = x['task']

    msg = [{
        "role": "user",
        "content": [
            {
                "type": x['data_type'],
                x['data_type']: os.path.join("/data/oss_bucket_0/yzl/Auto_Drive/AFile/datasets", x['path'])
            },
            {
                "type": "text",
                "text": QUESTION_TEMPLATE.format(original_information=information, original_problem=problem, original_task=task)
            }
        ]
    }]
    messages.append(msg)
    

final_output = []
start_idx = 0
if os.path.exists(OUTPUT_PATH):
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            existing = json.load(f)
            final_output = existing.get("results", [])
            start_idx = len(final_output)
            print(f"Resuming from sample index {start_idx}")
    except Exception as e:
        print(f"Error reading existing output file: {e}")


def extract_think(output_str):
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0).strip()  # 保留标签
    return ""


for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
    batch_messages = messages[i:i + BSZ]

    prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
    

    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)
        
        image_idx = 0
        video_idx = 0

        llm_inputs = []

        
        for idx, prompt in enumerate(prompts):
            mm_type = batch_messages[idx][0]['content'][0]['type']
            sample_mm_data = {}
            sample_video_kw = {}
            if mm_type == 'image':
                sample_mm_data["image"] = image_inputs[image_idx]
                image_idx += 1
            elif mm_type == 'video':
                sample_mm_data["video"] = video_inputs[video_idx]
                for key, value in video_kwargs.items():
                    sample_video_kw[key] = value[video_idx]
                video_idx += 1
                    
            
            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
                "mm_processor_kwargs": sample_video_kw,
            })
            

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        batch_output_text = [out.outputs[0].text for out in outputs]
        
    except Exception as e:
        print('error:', data[i]['path'])
        print('Exception:', e)
        batch_output_text = ['<answer>error</answer>'] * BSZ
        

    for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
        think_chain = extract_think(model_output)
        final_ans = extract_answer(model_output)

        sample["output"] = model_output.replace("\n", "").strip()
        sample["predict"] = final_ans.replace("\n", "").strip()
        final_output.append(sample)
        print("predict:", final_ans)

# 保存最终输出
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)