import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from analyze_qa import analyze

CKPT_HOME = "."
MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

if __name__ == "__main__":
    input_directory = "./test_videos"
    output_directory = "./output_qa"
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_path = f"{CKPT_HOME}/{model_name.split('/')[1]}"
    os.makedirs(output_directory, exist_ok=True)
    output_filename = f"{output_directory}/{model_name.split('/')[1]}.jsonl"
    print("=======")
    print(f"model_path: {model_path}")

    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(model_path)
    

    with open('video_perspective.json', 'rb') as f:
        data = json.load(f)
    
    output_dict = {}
    if os.path.exists(output_filename):
        with open(output_filename, 'r') as f:
            for line in f:
                output_dict.update(json.loads(line))

    for item in data:
        video_name = item["video_name"]
        if video_name in output_dict:
            continue
        video_path = os.path.join(input_directory, video_name+'.mp4')
        value_list = []
        for question in item['questions']:
            task_type = question['task_type']
            correct_answer = question['correct_answer']
            prompt = f"Carefully watch the video and pay attention to temporal dynamics in this video, focusing on the camera motions, actions, activities, and interactions. Based on your observations, select the best option that accurately addresses the question.\n{question['question']}\nYou can only response with the answer among {question['options']}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        
                        {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0,},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_text = output_text[0]
            
            if correct_answer.lower() in output_text.lower():
                judge = True
            else:
                judge = False
            
            value_list.append({'task_type':task_type, 'correct_answer':correct_answer, 'output':output_text, 'judge':judge})

        basename,_ = os.path.splitext(os.path.basename(video_path))
        print("=======")
        print(basename)
        print(value_list)
        
        new_item = {basename:value_list}

        with open(output_filename, 'a', encoding='utf-8') as output_file:
            json.dump(new_item, output_file, ensure_ascii=False)
            output_file.write('\n')
    
    analyze(output_filename)
