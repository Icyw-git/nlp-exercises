from transformers import AutoTokenizer
import datasets
from torch.utils.data import Dataset,DataLoader
from llm_LLaMA2 import ModelConfig, Transformer
import torch.nn.functional as F
import torch
import time
import swanlab
swanlab.login('smj4YZJHedo5rbWoy5DvT')






# 加载预训练的分词器
tokenizer=AutoTokenizer.from_pretrained('tokenizer')

#预训练数据准备
with open('./data/input.txt','r',encoding='utf-8') as f:
    text=f.read()

print(f'文本长度：{len(text)}')

tokens=tokenizer(text,return_tensors='pt')['input_ids'][0] #使用分词器将文本转换为输入ID，并返回一个PyTorch张量，选择第一个元素（因为返回的是一个批次的输入ID），得到一个一维张量，表示文本的分词结果

print(f'分词后的长度：{len(tokens)}')

max_length=512


num_blocks=tokens.numel() // max_length
tokens=tokens[:, :num_blocks*max_length] #截断多余的部分，使得总长度是max_length的整数倍
chunks=tokens.view(num_blocks,max_length) #将输入ID重新组织成块，每块的长度为max_length，得到一个二维张量，其中每行表示一个块

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

epochs=3
args=ModelConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=Transformer(args).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1)

for epoch in range(epochs):
    model.train()
    total_loss,total_tokens,start,end = 0,0,0,0
    start=time.time()
    for step,batch in enumerate(dataloader):
        optimizer.zero_grad()
        input_ids=batch['input_ids'][:,-1].to(device)
        labels=batch['labels'][:,1:].to(device)

        outputs=model(input_ids,labels=labels)

        loss=outputs.last_loss.mean()
        loss.backward()
        optimizer.step()
        total_tokens+=labels.numel()
        total_loss+=loss.item()*labels.numel()
    end=time.time()

    swanlab.log({
        'epoch': epoch,
        'final_loss': total_loss / total_tokens,
        'final_perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item(),
        'training_time': end - start
    })

    print(f'Epoch: {epoch}, Loss: {total_loss/total_tokens:.4f}, Time: {end-start:.2f}s, Perplexity: {torch.exp(torch.tensor(total_loss/total_tokens)):.2f}')


swanlab.init(
    project='my-awesome-project',
    experiment='llm-pretraining-demo',
    tags=['pretraining','transformer'],
    config={
        'epochs': epochs,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'model_config': args.__dict__
    }
)

