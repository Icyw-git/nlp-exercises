import json
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
import torch
from llm_LLaMA2 import ModelConfig,Transformer
import swanlab
import time
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)

tokenizer=AutoTokenizer.from_pretrained('tokenizer')

class SFTDataset(Dataset):
    def __init__(self,file_path,tokenizer,max_length):
        super().__init__()
        self.path=file_path
        self.tokenizer=tokenizer
        self.max_length=max_length
        with open(self.path,'r',encoding='utf-8') as f:
            self.data=json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        example=self.data[idx]
        instruction=example['instruction']
        input=example.get('input','') #使用get方法获取输入内容，如果没有输入字段，则默认为空字符串，这样可以确保在构建提示时不会出现错误，并且可以处理没有输入的情况。
        output=example['output']

        prompt=f'###User:\n{instruction}\n{input}\n\n###Assistant:\n{output}\n'
        answer=output

        prompt_ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        answer_ids=self.tokenizer.encode(answer,add_special_tokens=False)

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

dataset=SFTDataset('./data/alpaca_data_cleaned.json',tokenizer,256)


def collate_fn(batch,pad_id=tokenizer.eos_token_id,label_pad_id=-100):
    max_len=max(x['input_ids'].numel() for x in  batch)
    input_ids=[]
    labels=[]
    for x in batch:
        ids=x['input_ids']
        lab=x['labels']
        pad_len=max_len-len(ids)
        input_ids.append(torch.cat([ids,torch.full((pad_len,),pad_id,dtype=torch.long)]))
        labels.append(torch.cat([lab,torch.full((pad_len,),label_pad_id,dtype=torch.long)]))

    return {'input_ids': torch.stack(input_ids), 'labels': torch.stack(labels)}

dataloader=DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=collate_fn)

args=ModelConfig()

swanlab.init(
    project='my-awesome-project',
    experiment='llm-sft-demo',
    tags=['sft','transformer'],
    config={
        'epochs': 4,
        'batch_size': 4,
        'learning_rate': 3e-4,}
)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Transformer(args).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.01)

epochs=4

for epoch in range(epochs):
    model.train()
    total_loss,total_tokens,start,end=0,0,0,0
    start=time.time()
    for step,batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids=batch['input_ids'][:,:-1].to(device)
        labels=batch['labels'][:,1:].to(device)

        outputs=model(input_ids,labels=labels)

        loss=outputs.last_loss.mean()
        loss.backward()
        optimizer.step()

        total_tokens+=labels.numel()
        total_loss+=loss.item()*labels.numel()
    end=time.time()
    epoch_loss=total_loss/total_tokens
    epoch_time=end-start
    swanlab.log({
        'epoch_loss': epoch_loss,
        'epoch_time': epoch_time
    })
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f} seconds')


swanlab.finish()
print('训练完成！')
