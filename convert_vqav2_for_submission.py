import os
import argparse
import json
import re

import sys
sys.path.append('../LLaVA')
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor


def truncate_string(input_str):
    # 定义匹配 '\n', '.', ',', 'Question' 的正则表达式
    pattern = r'[\n.,]|Question'
    
    # 使用正则表达式进行分割
    result = re.split(pattern, input_str, maxsplit=1)
    
    # 返回分割后的第一个部分（截断后的字符串）
    return result[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="./playground/data/eval/vqav2")
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    src = os.path.join(args.dir, 'answers', args.split, args.ckpt, 'merge.jsonl')
    test_split = os.path.join(args.dir, f'{args.split}.jsonl')
    dst = os.path.join(args.dir, 'answers_upload', args.split, f'{args.ckpt}.json')
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    results = []
    error_line = 0
    for line_idx, line in enumerate(open(src)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(test_split)]
    split_ids = set([x['question_id'] for x in test_split])

    print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

    all_answers = []

    answer_processor = EvalAIAnswerProcessor()

    for x in test_split:
        if x['question_id'] not in results:
            all_answers.append({
                'question_id': x['question_id'],
                'answer': ''
            })
        else:
            ans = results[x['question_id']]
            # ans = ans.removeprefix('\nAnswer: ') # FIXME: Xgen-MM seems to usually output \nAnswer: at first.
            ans = truncate_string(ans)
            all_answers.append({
                'question_id': x['question_id'],
                'answer': answer_processor(ans)
            })

    with open(dst, 'w') as f:
        json.dump(all_answers, open(dst, 'w'))
