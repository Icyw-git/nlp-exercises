import datasets
import torch

from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
import math
import warnings
import platform
import argparse
import pandas as pd
import torch.optim as optim
from contextlib import nullcontext
import os
import time
from llm_LLaMA2 import ModelConfig, Transformer

import swanlab

#忽略警告信息
warnings.fliterwarnings('ignore')


def Logger(content):
    print(content)






# 加载预训练的分词器
tokenizer=AutoTokenizer.from_pretrained('tokenizer')

#预训练数据准备
with open('./data/input.txt','r',encoding='utf-8') as f:
    text=f.read()

print(f'文本长度：{len(text)}')

tokens=tokenizer(text,return_tensors='pt')['input_ids'][0] #使用分词器将文本转换为输入ID，并返回一个PyTorch张量，选择第一个元素（因为返回的是一个批次的输入ID），得到一个一维张量，表示文本的分词结果

print(f'分词后的长度：{len(tokens)}')

max_length=512
chunks=[]

for i in range(0,len(tokens),max_length):
    chunk=tokens[i:i+max_length]
    chunks.append(chunk)
print(f'分块数量：{len(chunks)}')

class Pretraindataset(Dataset):
    def __init__(self,chunks):
        self.chunks=chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self,idx):
        chunk=self.chunks[idx]

        return {
            'input_ids': chunk,
            'labels': chunk.clone() # 克隆输入ID作为标签，确保输入和标签是相同的
        }

dataset = Pretraindataset(chunks)
first_data=dataset[0]
print(f'输入ID：{first_data["input_ids"]}')
print(f'标签：{first_data["labels"]}')

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
for batch in dataloader:
    print(f'批次输入ID：{batch["input_ids"]}')
    print(f'批次标签：{batch["labels"]}')
    break





def get_lr(it,all):
    #it:当前迭代次数，all:总迭代次数

    warmup_iters=args.warmup_iters
    lr_decay_iters=all
    min_lr=args.learning_rate/10

    if it<warmup_iters:
        return args.learning_rate *it/warmup_iters
    if it>lr_decay_iters:
        return min_lr

    decay_ratio=(it-warmup_iters) /(lr_decay_iters-warmup_iters)
    assert 0<=decay_ratio<=1

    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(args.learning_rate-min_lr)

def train_epoch(epoch):

    start_time=time.time()
    for step,(X,Y,loss_mask) in enumerate(dataloader):
        X=X.to(args.device)
        Y=Y.to(args.device)
        loss_mask=loss_mask.to(args.device)

        lr=get_lr(epoch*iter_per_epoch+step,args.epochs*iter_per_epoch)
        for param_group in optimizer.param_groups:

            param_group['lr']=lr

        with ctx:
            out=model(X,Y)
            loss =out.last_loss/args.accumulation_steps
            loss_mask=loss_mask.view(-1)
            loss=torch.sum(loss*loss_mask)/loss_mask.sum()

        scaler.scale(loss).backward()

        if(step+1) %args.accumulation_step==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step %args.log_interval==0:
            spend_time=time.time()-start_time

            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch time:{}min;'.format(
                    epoch+1,
                    args.epochs,
                    step,
                    iter_pre_epoch,
                    loss.item()*args.accumulation_steps,
                    optimizer,param_groups[-1]['lr'],
                    spend_time /(step+1)* iter_per_epoch // 60-spend_time //60)

                )

            if args.use_swanlab:
                swanlab.login('smj4YZJHedo5rbWoy5DvT')
                swanlab.log(
                    {
                        'loss':loss.item()*args.accumulation_steps,
                        'lr':optimizer.param_groups[-1]['lr']
                    }
                )

        if (step+1)%args.save_interval==0:
            model.eval()

            ckp=f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm.config.vocab_size}.pth'

            state_dict=model.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict,ckp)
            model.train()

        if(step+1)%20000==0:
            model.eval()

            ckp=f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm.config.vocab_size}_step{step+1}.pth'

            state_dict=model.module.state_dict() if isinstance(model,torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict,ckp)
            model.train()




