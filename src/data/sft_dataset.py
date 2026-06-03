from torch.utils.data import Dataset

import json
import torch

from src.data.template import build_prompt



class SFTDataset(Dataset):
    def __init__(self, data_file: list, tokenizer, max_length: int, template: str, from_list=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        self.data = data_file
        if not from_list:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        example = self.data[idx]
        instruction = example['instruction']
        input = example.get('input', '')
        output = example['output']

        prompt = build_prompt(instruction, input, self.template)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(output, add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(self.tokenizer.eos_token_id)

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        assert len(input_ids) == len(labels)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
