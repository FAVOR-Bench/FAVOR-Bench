'''
Calculate the accuracy of each subtask
'''
import json

def analyze(input_file):
    all_dict = {
        "ALL":[0,8184],
        "AS":[0,2637],
        "HAC":[0,1541],
        "SAD":[0,1662],
        "MAD":[0,1205],
        "CM":[0,1075],
        "NSA":[0,64]
    }
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            key = next(iter(item))
            for question in item[key]:
                task_type = question["task_type"]
                if question["judge"] == True:
                    all_dict[task_type][0] += 1
                    all_dict["ALL"][0] += 1
            a=1

    scores1 = {key: round(value[0] / value[1] * 100, 2) for key, value in all_dict.items()}
    scores = [round(value[0] / value[1] * 100, 2) for value in all_dict.values()]
    formatted_output = " & ".join([f"{score}" for score in scores])
    print(input_file)
    for key, score in scores1.items():
        print(f"{key}: {score}%")
    print(formatted_output)

if __name__ == "__main__":
    analyze("output_qa/Qwen2.5-VL-7B-Instruct.jsonl")
