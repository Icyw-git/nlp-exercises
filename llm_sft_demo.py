import json
import datasets
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained('tokenizer')

class SFTDataset(Dataset):
    def __init__(self,file_path,tokenizer,max_length):
        super().__init__()
        self.path=file_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        self.data=[json.load(f) for f in open(self.path,'r',encoding='uft-8')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        example=self.data[idx]
        instruction=example['instruction']
        input=example.get('input','') #使用get方法获取输入内容，如果没有输入字段，则默认为空字符串，这样可以确保在构建提示时不会出现错误，并且可以处理没有输入的情况。
        output=example['output']

        prompt=f'###User:\n{instruction}\n{input}\n\n###Assistant:\n{output}\n'
        answer=output

        prompt_ids=self.tokenizer.encode(prompt,add_special_tokens=False)['input_ids']
        answer_ids=self.tokenizer.encode(answer,add_special_tokens=False)['input_ids']

        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(self.tokenizer.eos_token_id)

        input_ids=prompt_ids+answer_ids
        if len(input_ids)>self.max_length:
            input_ids=input_ids[:self.max_length]

        labels=[-100]*len(prompt_ids)+answer_ids
        if len(labels)>self.max_length:
            labels=labels[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids,dtype=torch.long),
            'labels': torch.tensor(labels,dtype=torch.long),
        }




def collate_fn(batch):
