import argparse
import os
import math
from tqdm import tqdm
import shortuuid
import copy

from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image
import json
import PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_pretrained_model():
    processor = AutoProcessor.from_pretrained('HuggingFaceM4/idefics2-8b-base')
    model = AutoModelForVision2Seq.from_pretrained('HuggingFaceM4/idefics2-8b-base', torch_dtype=torch.float16)

    return processor, model


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, processor):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor

        # prepare in-context-examples
        with open('vqav2_few_shots.json', 'r') as f:
            incontext_data = json.load(f)
        self.context_images, self.context_text = [], ''
        path_prefix = 'playground/data/eval/vqav2/train2014'

        for idx, example in enumerate(incontext_data):
            img = PIL.Image.open(os.path.join(path_prefix, incontext_data[example]['image_path']))
            self.context_images.append(img)

            instruction = incontext_data[example]['instruction']
            answer = incontext_data[example]['output']
            example_text = f'<image>Question: {instruction} Give a very brief answer. Answer: {answer}\n'
            self.context_text += example_text

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line['image']
        qs = line['text']
        qs = qs.split('\n')[0] # Remove \nAnswer the question using a single word or phrase.
        prompts = f'<image>Question: {qs} Give a very brief answer. Answer: '
        cont_prompts = copy.deepcopy(self.context_text)
        cont_prompts += prompts

        img = PIL.Image.open(os.path.join(self.image_folder, image_file))
        images = img
        cont_images = copy.deepcopy(self.context_images)
        cont_images.append(images)

        # import pdb; pdb.set_trace()

        inputs = self.processor(text=[cont_prompts], images=[cont_images], padding=True, return_tensors='pt')
        # import pdb; pdb.set_trace()
        # inputs = {name: tensor.to('cuda') for name, tensor in inputs.items()}

        return inputs
    
    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, image_folder, processor, batch_size=1, num_workers=4):
    assert batch_size == 1, 'batch_size must be 1'
    dataset = CustomDataset(questions, image_folder, processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return data_loader


def eval_model(args):
    processor, model = load_pretrained_model()
    model = model.to('cuda')
    # tokenizer.padding_side = "left"

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), 'r')]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, 'w')

    data_loader = create_data_loader(questions, args.image_folder, processor)

    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line['question_id']
        cur_prompt = line['text']

        # import pdb; pdb.set_trace()
        inputs['pixel_values'] = inputs['pixel_values'].squeeze(0).to('cuda')
        inputs['pixel_attention_mask'] = inputs['pixel_attention_mask'].squeeze(0).to('cuda')
        inputs['input_ids'] = inputs['input_ids'].squeeze(0).to('cuda')
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0).to('cuda')

        with torch.cuda.amp.autocast(dtype=torch.float16):
            generated_ids = model.generate(**inputs, max_new_tokens=5)
        
        generated_texts = processor.decode(generated_ids[-1], skip_special_tokens=True)
        generated_texts = generated_texts.split('Answer:')[-1].strip()
        # import pdb; pdb.set_trace()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": generated_texts,
                                   "answer_id": ans_id,
                                   "model_id": 'idefics2-base-8b',
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
