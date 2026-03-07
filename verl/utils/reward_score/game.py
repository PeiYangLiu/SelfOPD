import re

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    if "####" in solution_str:
        solution = solution_str.split("####")[-1].strip().split(',')
        for i in range(len(solution)):
            solution[i] = solution[i].strip()
        return solution

def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for Game.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer in ground_truth:
            return score
        else:
            return format_score