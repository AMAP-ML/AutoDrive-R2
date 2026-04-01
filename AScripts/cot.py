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


MODEL_PATH = "/mnt/xmap_nas_alg/yzl/Amodel/Qwen2.5-VL-72B-Instruct"
BSZ = 50


llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    # max_model_len = 8192,
    gpu_memory_utilization=0.9,
    # limit_mm_per_prompt={"image": 10, "video": 10},
)

# Add stop tokens for structured output
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.8,
    max_tokens=4096,
    stop_token_ids=[]
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

for dataset_name in ['your_data_name']:

    OUTPUT_PATH = "/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AData/sft_cot.json"
    PROMPT_PATH = "/mnt/xmap_nas_alg/yzl/Auto_Drive_Github/AData/sft.json"
    
    data = []
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
You are given an image, a driving-related question, and its answer. Generate a **four-stage reasoning process** with explicit mathematical modeling and self-validation.
Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions
It's encouraged to include self-reflection or verification in the reasoning process. 

### **Input Format:**  
- **System Instructions**: {original_task}  
- **Past Vehicle Status**: {original_information}  
- **Prediction Task**: {original_problem}  
- **Answer**: {original_solution}  

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
"""
    )

    messages = []
    for x in data:
        information = x['information']
        problem = x['problem']
        solution = x['solution']
        task = x['task']

        msg = [{
            "role": "user",
            "content": [
                {
                    "type": x['data_type'],
                    x['data_type']: os.path.join("/mnt/xmap_nas_alg/yzl/Auto_Drive/AFile/datasets", x['path'])
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(original_information=information, original_problem=problem, original_solution=solution, original_task=task)
                }
            ]
        }]
        messages.append(msg)

    # For resume
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
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)

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
            batch_output_text = ['<answer>error</answer>'] * BSZ
            

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            # sample["answer"] = final_ans
            # q_type = sample.get("problem_type", "")
            # sample["reward"] = reward_fn(sample, model_output, q_type)
            # sample['select'] = True if sample["reward"] > 0.6 else False
            # if think_chain:
            #     sample["process"] = f"<think>{think_chain}</think>"

            # sample["process"] = f"<think>{think_chain}</think>"
            sample["process"] = model_output
            final_output.append(sample)
            
        
        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    print(f"Results saved to {OUTPUT_PATH}")
