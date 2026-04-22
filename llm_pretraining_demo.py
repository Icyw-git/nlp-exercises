# -*- coding: utf-8 -*-


from transformers import AutoTokenizer
from torch.utils.data import Dataset,DataLoader
from llm_LLaMA2 import ModelConfig, Transformer
import torch
import time
import swanlab
from dotenv import load_dotenv
import os
import numpy as np
import random

#设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


#从环境中加载swanlab的密钥，并登录swanlab平台，以便在后续的训练过程中记录和追踪实验数据。
load_dotenv()
api_key = os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)

args=ModelConfig() #创建modelconfig对象，包含模型的配置参数，例如层数、隐藏维度、注意力头数等，这些参数将用于定义和训练Transformer模型。
swanlab.init(   #在使用swanlab之前先进行init
    project='my-awesome-project',
    experiment='llm-pretraining-demo',
    tags=['pretraining','transformer'],
    config={
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'model_config': args.__dict__  #__dict__属性是Python对象的一个内置属性，它返回一个字典，包含对象的所有属性和它们的值。在这里，args是一个ModelConfig对象，args.__dict__将返回一个字典，其中包含ModelConfig对象的所有配置参数及其对应的值，这些参数将被记录到Swanlab中，以便在实验中进行追踪和分析。
    }
)






# 加载预训练的分词器
tokenizer=AutoTokenizer.from_pretrained('tokenizer')

#预训练数据准备
with open('./data/input.txt','r',encoding='utf-8') as f:
    text=f.read()

print(f'文本长度：{len(text)}')

tokens=tokenizer(text,return_tensors='pt')['input_ids'][0] #使用分词器将文本转换为输入ID，并返回一个PyTorch张量，选择第一个元素（因为返回的是一个批次的输入ID），得到一个一维张量，表示文本的分词结果

print(f'分词后的长度：{len(tokens)}')

max_length=512

#去除最后一块多余的部分，保证总长度是max_length的整数倍，这样可以方便后续的分块处理，避免最后一块长度不足的问题
num_blocks=tokens.numel()//max_length
train_blocks=int(tokens.numel() // max_length *0.9)
train_tokens=tokens[:train_blocks*max_length] #截断多余的部分，使得总长度是max_length的整数倍
train_chunks=train_tokens.view(train_blocks,max_length) #将输入ID重新组织成块，每块的长度为max_length，得到一个二维张量，其中每行表示一个块

valid_blocks=num_blocks-train_blocks
valid_tokens=tokens[train_blocks*max_length:(train_blocks + valid_blocks) * max_length]
valid_chunks=valid_tokens.view(valid_blocks,max_length)


class Pretraindataset(Dataset):
    def __init__(self,chunks):
        self.chunks=chunks

    def __len__(self):  #__len__方法返回数据集的大小，即块的数量，这对于PyTorch的数据加载器来说是必要的，以便正确地迭代数据集。
        return len(self.chunks)

    def __getitem__(self,idx): #__getitem__方法根据给定的索引idx返回数据集中的一个样本，这里返回的是一个字典，包含输入ID和标签。输入ID是从块中获取的，而标签是输入ID的克隆，确保输入和标签是相同的。这种设计适用于语言模型的预训练任务，其中模型需要预测下一个标记，因此输入和标签是相同的。
        chunk=self.chunks[idx]

        return {
            'input_ids': chunk,
            'labels': chunk.clone() # 克隆输入ID作为标签，确保输入和标签是相同的
        }

traindataset = Pretraindataset(train_chunks)
first_data=traindataset[0]
print(f'输入ID：{first_data["input_ids"]}')
print(f'标签：{first_data["labels"]}')



trainloader = DataLoader(traindataset, batch_size=4, shuffle=True)
for batch in trainloader:  #这里的batch包含input_ids和labels两个键，分别对应输入ID和标签。通过迭代dataloader，我们可以获取每个批次的数据，并在训练循环中使用这些数据进行模型的训练。
    print(f'批次输入ID：{batch["input_ids"]}')
    print(f'批次标签：{batch["labels"]}')
    break


valid_dataset = Pretraindataset(valid_chunks)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)



def eval_on_valid_set(model,valid_loader):
    model.eval()

    with torch.no_grad():
        total_loss,total_tokens=0,0
        for batch in valid_loader:
            inputs=batch['input_ids'][:,:-1].to(device)
            labels=batch['labels'][:,1:].to(device)

            outputs=model(inputs,labels)
            loss=outputs.last_loss.mean()
            total_loss+=loss.item()*labels.numel()
            total_tokens+=labels.numel()
        return total_loss/total_tokens







epochs=3
args=ModelConfig()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #在模型定义的时候不使用这条语句，而是在训练循环中使用，这样可以确保模型和数据都被正确地移动到GPU上（如果可用）。在模型定义时，模型的参数默认是在CPU上创建的，如果直接将模型定义放在GPU上，可能会导致一些问题，例如在某些环境中可能无法正确识别GPU设备，或者在模型定义时就占用GPU资源，导致后续的训练过程出现问题。因此，建议在训练循环中使用这条语句来动态地检测和使用GPU设备，以确保模型和数据都能正确地利用GPU加速训练。
model=Transformer(args).to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),weight_decay=0.1)  #betas是Adam优化器的超参数，控制一阶矩估计和二阶矩估计的指数衰减率，weight_decay是权重衰减系数，用于正则化模型，防止过拟合

best_loss=float('inf')
patience=10
pat_counter=0





for epoch in range(epochs):
    model.train()
    total_loss,total_tokens,start,end = 0,0,0,0
    start=time.time()
    for step,batch in enumerate(trainloader):
        optimizer.zero_grad()
        input_ids=batch['input_ids'][:,:-1].to(device) #使用shifted input_ids作为模型的输入，shifted input_ids是将原始输入ID向右移动一个位置得到的，这样模型在训练过程中就可以学习预测下一个标记。具体来说，input_ids[:, :-1]表示去掉每个序列的最后一个标记，而labels=batch['labels'][:, 1:]表示去掉每个序列的第一个标记，这样输入和标签就对齐了，模型可以学习从输入预测标签。
        labels=batch['labels'][:,1:].to(device)

        outputs=model(input_ids,labels=labels)

        loss=outputs.last_loss.mean() #last_loss是模型输出每个位置上的损失值，mean()方法计算这些损失值的平均值，得到一个标量损失值，这个损失值用于反向传播和优化模型参数。
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0) #梯度裁剪，防止梯度爆炸，clip_grad_norm_函数用于对模型参数的梯度进行裁剪，model.parameters()返回模型的所有参数，1.0是裁剪的最大范数值，这样可以确保在反向传播过程中，梯度的范数不会超过1.0，从而稳定训练过程。
        optimizer.step()
        total_tokens += labels.numel()  # token数量通过labels.numel()计算
        total_loss += loss.item() * labels.numel()

        if step % 100 ==0:  #每100步保存一次模型的检查点，checkpoint是模型训练过程中的一个快照，包含了模型的当前状态、优化器的状态、当前的训练轮数和步数，以及随机数生成器的状态等信息，这些信息可以用于在训练过程中断后恢复训练，或者在训练完成后进行模型评估和推理。
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'rng_state':torch.get_rng_state(),

                },f'checkpoint_s{step}_e{epoch}.pth'
            )

    end = time.time()

    val_loss=eval_on_valid_set(model,valid_loader)
    swanlab.log({
        'valid_loss': val_loss,
        'ppl': torch.exp(val_loss).item()
    })

    if val_loss < best_loss:
        best_loss=val_loss
        pat_counter=0
    else:
        pat_counter+=1
        if pat_counter>patience:
            print('Early stopping triggered!')
            break



    swanlab.log({
        'epoch': epoch,
        'train_loss': total_loss / total_tokens,
        'train_perplexity': torch.exp(torch.tensor(total_loss / total_tokens)).item(),
        'train_time': end - start
    })







swanlab.finish()
print('训练完成！')

print("tokens:", tokens.numel())
print("steps_per_epoch:", len(trainloader))


print(tokenizer.decode([0]))