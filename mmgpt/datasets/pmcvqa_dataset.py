import copy
import json
import os
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer

TEMPLATE = {
    "description": "Template used by PMC VQA.",
    "prompt_choice": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Input:\n{options}\n\n### Response:\n",
    "prompt_qa": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Image:\n{image}\n\n### Instruction:\n{question}\n\n### Response:\n",
    "response_split": "### Response:",
}

class PMCVQAPrompter:
    def __call__(self, question, options=None):
        if options:
            options = ", ".join(options)
            res = TEMPLATE["prompt_choice"].format(image="<image>", question=question, options=options)
        else:
            res = TEMPLATE["prompt_qa"].format(image="<image>", question=question)
        return res

    def get_response(self, output: str) -> str:
        return output.split(TEMPLATE["response_split"])[-1].strip()

class PMCVQADataset(Dataset):
    """
    DATASET FORMAT:
    Figure_path	PMC1064097_F1.jpg
    Question	What is the uptake pattern in the breast?
    Answer	Focal uptake pattern
    Choice A	A:Diffuse uptake pattern
    Choice B	B:Focal uptake pattern
    Choice C	C:No uptake pattern
    Choice D	D:Cannot determine from the information given
    Answer_label	B
    """
    def __init__(
        self,
        tokenizer,
        vis_processor=None,
        vis_root=None,
        ann_path="",
        add_eos=True,
        ignore_instruction=True,
        sample_image=False,
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        assert tokenizer.add_eos_token is False, "tokenizer should not add eos token by default"
        self.tokenizer: AutoTokenizer = tokenizer
        self.vis_root = vis_root

        self.annotation = self.load_annotation(ann_path)

        self.vis_processor = vis_processor
        self.__add_instance_ids()
        self.option_prob = 0.5
        self.prompter = PMCVQAPrompter()
        self.add_eos = add_eos
        self.ignore_instruction = ignore_instruction
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, idx):
        ann = self.annotation[idx]
        text = self.process_text(ann)
        image = self.process_image(ann)
        res = self.tokenize(text)
        res.update(image=image)
        res.update(text)
        return res
    
    def load_annotation(self, ann_path):
        # convert csv to json
        ann = []
        with open(ann_path, "r") as f:
            for line in f:
                if line.startswith("Figure_path"): continue
                line = line.strip().split(",")
                ann.append(dict(zip(["Figure_path", "Question", "Answer", "Choice A", "Choice B", "Choice C", "Choice D", "Answer_label"], line)))
        return ann

    def tokenize(self, text):
        res = self.tokenizer(
            text["instruction"] + text["answer"],
            return_tensors=None,
            padding="do_not_pad",
            truncation=True,
            max_length=512,
        )

        # manually add eos token
        if res["input_ids"][-1] != self.tokenizer.eos_token_id and len(res["input_ids"]) < 512 and self.add_eos:
            res["input_ids"].append(self.tokenizer.eos_token_id)
            res["attention_mask"].append(1)
        labels = copy.deepcopy(res["input_ids"])
        # ignore instruction_token
        if self.ignore_instruction:
            instruction_token = self.tokenizer(
                text["instruction"], return_tensors=None, padding="do_not_pad", truncation=True, max_length=512
            )
            labels = [-100] * len(instruction_token["input_ids"]) + labels[len(instruction_token["input_ids"]) :]

        res.update(labels=labels)
        return res
    
    def process_image(self, ann):
        image_path = os.path.join(self.vis_root, ann["Figure_path"])
        image = Image.open(image_path)
        if self.vis_processor:
            image = self.vis_processor(image)
        return image
    
    def process_text(self, ann):
        question = ann["Question"]
        true_answer = ann["Answer"]
        is_option = random.random() < self.option_prob
        if is_option:
            instruction = self.prompter(question, [ann["Choice A"], ann["Choice B"], ann["Choice C"], ann["Choice D"]])
        else:
            instruction = self.prompter(question)
        return dict(instruction=instruction, answer=true_answer)
        

    def __add_instance_ids(self, key="id"):
        for idx, ann in enumerate(self.annotation):
             ann[key] = str(idx)

    def collater(self, samples):
        image_list, question_list, answer_list, input_id_list, attention_mask_list, labels_list = [], [], [], [], [], []
        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["instruction"])
            answer_list.append(sample["answer"])
            input_id_list.append(sample["input_ids"])
            attention_mask_list.append(sample["attention_mask"])
            labels_list.append(sample["labels"])
        
        max_label_length = max([len(l) for l in labels_list])
        padding_side = self.tokenizer.padding_side
        padded_labels = []
        for l in labels_list:
            remainder = [-100] * (max_label_length - len(l))
            if isinstance(l, list):
                l = l + remainder if padding_side == "right" else remainder + l
            elif padding_side == "right":
                l = np.concatenate([l, remainder]).astype(np.int64)
            else:
                l = np.concatenate([remainder, l]).astype(np.int64)
            padded_labels.append(l)

        padded_samples = self.tokenizer.pad(
            {"input_ids": input_id_list, "attention_mask": attention_mask_list, "labels": padded_labels},
            return_tensors="pt",
            padding="longest",
        )

        labels = padded_samples["labels"]
        media_token_id = self.tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, 0] = -100
        labels[labels == media_token_id] = -100
        return {
            "image": torch.stack(image_list, dim=0),
            "input_ids": padded_samples["input_ids"],
            "attention_mask": padded_samples["attention_mask"],
            "labels": labels,
            "instruction": question_list,
            "answer": answer_list,
        }
    
