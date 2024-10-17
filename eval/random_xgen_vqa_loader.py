import argparse
import os
import math
from tqdm import tqdm
import shortuuid
import random

from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
import json
import PIL
import torch
from torch.utils.data import Dataset, DataLoader


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_pretrained_model():
    model_name_or_path = "Salesforce/xgen-mm-phi3-mini-base-r-v1.5"
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True, legacy=False)
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = model.update_special_tokens(tokenizer)

    return tokenizer, model, image_processor


def apply_prompt_template(prompt, num_images=1, num_tokens_per_vis = 128, in_context=False, output=None):
    """
    num_tokens_per_vis: model.vlm.num_tokens_per_vis
    """
    placeholder_image_tokens = "<image placeholder>" * (num_tokens_per_vis - 1)
    if in_context:
        formatted_prompt = f"<image>{placeholder_image_tokens}" + f"{prompt}" + f"{output}" + "<|endofchunk|>"
    else:
        formatted_prompt = f"<image>{placeholder_image_tokens}"*num_images + f"{prompt}"
    return formatted_prompt


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        # prepare in-context-examples
        # with open('vqav2_few_shots.json', 'r') as f:
        #     incontext_data = json.load(f)
        # self.context_images, self.context_text = [], ''
        self.path_prefix = 'playground/data/eval/vqav2/train2014'
        
        # for example in incontext_data:
        #     img = PIL.Image.open(os.path.join(path_prefix, incontext_data[example]['image_path']))
        #     instruction = incontext_data[example]['instruction']
        #     instruction += '\nAnswer the question using a single word or phrase.'
        #     example_text = apply_prompt_template(prompt=instruction, in_context=True, output=incontext_data[example]['output'])
        #     self.context_images.append(img)
        #     self.context_text += (example_text)

        vqav2_train_ques_path = 'vqav2_train_ques_and_anno/v2_OpenEnded_mscoco_train2014_questions.json'
        vqav2_train_anno_path = 'vqav2_train_ques_and_anno/v2_mscoco_train2014_annotations.json'

        self.train_questions = json.load(open(vqav2_train_ques_path, 'r'))['questions']
        self.train_annotations = json.load(open(vqav2_train_anno_path, 'r'))['annotations']

    def sample_demos(self, num_samples=8):
        context_images, context_text = [], ''

        for idx, (ques, anno) in enumerate(random.sample(list(zip(self.train_questions, self.train_annotations)), num_samples)):
            image = "COCO_train2014_{:012d}.jpg".format(int(ques['image_id']))
            # sampled_list.append((ques['question'], anno['answers'][0]['answer'], image))
            instruction = ques['question']
            instruction += '\nAnswer the question using a single word or phrase.'
            # import pdb; pdb.set_trace()
            example_text = apply_prompt_template(prompt=instruction, in_context=True, output=anno['answers'][0]['answer'])
            context_text += (example_text)

            img = PIL.Image.open(os.path.join(self.path_prefix, image))
            context_images.append(img)

        return context_text, context_images

    def __getitem__(self, index):
        context_text, context_images = self.sample_demos()

        line = self.questions[index]
        image_file = line['image']
        qs = line['text']
        # qs = qs.split('\n')[0] # Remove \nAnswer the question using a single word or phrase.
        prompt = apply_prompt_template(qs)
        batch_text = context_text + prompt
        import pdb; pdb.set_trace()
        language_inputs = self.tokenizer([batch_text], return_tensors="pt")

        img = PIL.Image.open(os.path.join(self.image_folder, image_file))
        batch_images = context_images + [img]
        inputs = self.image_processor(batch_images, return_tensors="pt")

        inputs.update(language_inputs)
        # inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}

        return inputs
    
    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, image_folder, tokenizer, image_processor, batch_size=1, num_workers=0):
    assert batch_size == 1, 'batch_size must be 1'
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return data_loader


def eval_model(args):
    tokenizer, model, image_processor = load_pretrained_model()
    model = model.to('cuda')
    tokenizer.padding_side = "left"

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), 'r')]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, 'w')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor)

    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line['question_id']
        cur_prompt = line['text']

        # import pdb; pdb.set_trace()
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0).to('cuda')
        inputs['input_ids'] = inputs['input_ids'].squeeze(0).to('cuda')
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0).to('cuda')

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # generated_text = model.generate(**inputs,
            #                                 pad_token_id=tokenizer.pad_token_id,
            #                                 do_sample=False, max_new_tokens=256, top_p=None, num_beams=1,
            #                                 length_penalty=1.0, repetition_penalty=2.0)
            generated_text = model.generate(**inputs, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            do_sample=False, max_new_tokens=5, top_p=None, num_beams=1,
                                            length_penalty=1.0, repetition_penalty=2.0)

        prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        # import pdb; pdb.set_trace()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": prediction,
                                   "answer_id": ans_id,
                                   "model_id": 'xGEN-MM-base',
                                   "metadata": {}}) + "\n")
        
        # import pdb; pdb.set_trace()
    
    ans_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--question-file', type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)

    args = parser.parse_args()

    eval_model(args)
