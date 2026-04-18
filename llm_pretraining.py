import datasets
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader

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

