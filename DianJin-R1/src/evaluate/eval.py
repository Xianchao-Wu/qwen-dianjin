import re
import json
import argparse

def eval_finqa(path):
    correct_cnt = 0
    total_cnt = 0
    with open(path) as f:
        for line in f:
            total_cnt += 1
            line = line.strip()
            item = json.loads(line)
            # gpt-4o评估结果
            response = item['output']
            if "boxed{{1}}" in response:
                correct_cnt += 1

    print("=" * 20 + " Accuracy " + "=" * 20)
    print(correct_cnt)
    print(total_cnt)
    print(correct_cnt / total_cnt)

def eval_cflue(path):
    box_pattern = r"\\boxed\{(.*?)\}"
    data = json.load(open(path, "r"))
    choices = "ABCDEFG"
    cnt = 0
    for item in data:
        # 标准答案
        answer = item["answer"]
        # 预测结果
        output = item["output"]
        if output is None:
            print("=" * 20 + " None " + "=" * 20)
            continue
        matches = re.findall(box_pattern, output, re.DOTALL)
        if len(matches) == 0:
            print("=" * 20 + " Wrong Format " + "=" * 20)
            print(item["instruction"])
            print(answer)
            continue
        else:
            pred = ""
            for c in choices:
                if c in matches[-1]:
                    pred += c
            if answer == pred:
                cnt += 1
            else:
                print("=" * 20 + " Wrong Answer " + "=" * 20)
                print(pred)
                print(answer)

    print("=" * 20 + " Accuracy " + "=" * 20)
    print(cnt)
    print(len(data))
    print(cnt / len(data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str)
    args = parser.parse_args()
    output_path = args.result_path
    eval_cflue(output_path)
    # eval_finqa(output_path)