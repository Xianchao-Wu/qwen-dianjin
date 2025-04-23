import re


def get_choices(text):
    choices = "ABCDEFG"
    answer = ""
    for c in choices:
        if c in text:
            answer += c
    return answer


def extract_answer(solution_str):
    box_pattern = r"\\boxed\{(.*?)\}"
    think_pattern = r"<think>(.*?)</think>"
    answer_pattern = r"<answer>(.*?)</answer>"

    think_matches = re.findall(think_pattern, solution_str, re.DOTALL)
    if len(think_matches) != 1:
        return None

    answer_matches = re.findall(answer_pattern, solution_str, re.DOTALL)
    if len(answer_matches) != 1:
        return None

    box_matches = re.findall(box_pattern, answer_matches[0], re.DOTALL)
    if len(box_matches) == 0:
        return None

    return get_choices(box_matches[-1])


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    else:
        gt = get_choices(ground_truth)
        if answer == gt:
            return 1.0
        else:
            return 0.0