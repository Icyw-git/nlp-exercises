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
            self.data=json.load(f) #json.load和json.loads的区别在于前者用于从文件中读取JSON数据并解析成Python对象，而后者用于从字符串中解析JSON数据。这里使用json.load是因为数据存储在一个JSON文件中，需要从文件中读取并解析成Python对象，以便后续的数据处理和模型训练。

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        example=self.data[idx]
        instruction=example['instruction']
        input=example.get('input','') #使用get方法获取输入内容，如果没有输入字段，则默认为空字符串，这样可以确保在构建提示时不会出现错误，并且可以处理没有输入的情况。
        output=example['output']

        prompt=f'###User:\n{instruction}\n{input}\n\n###Assistant:\n{output}\n'  #拼接成prompt字段，包含用户的指令、输入和助手的回答，这样可以为模型提供足够的上下文信息，以便进行监督学习微调。使用特定的格式（例如###User:和###Assistant:）可以帮助模型更好地理解不同角色之间的关系，从而提高生成回答的质量。
        answer=output

        prompt_ids=self.tokenizer.encode(prompt,add_special_tokens=False) #解码成ids，返回的数据类型是List,和tokenizer方法的区别是tokenizer方法返回一个字典，包含输入ID、注意力掩码等信息，返回的是tensor类型。

        answer_ids=self.tokenizer.encode(answer,add_special_tokens=False)

        if self.tokenizer.eos_token_id is not None:
            answer_ids.append(self.tokenizer.eos_token_id) #选择是否添加结束标记，如果分词器定义了结束标记（eos_token_id），则将其添加到答案ID的末尾，以便模型在生成回答时知道何时停止生成。

        input_ids=prompt_ids+answer_ids
        if len(input_ids)>self.max_length:
            input_ids=input_ids[-self.max_length:] #若超过最大序列长度，就截断输出id，保留最后的max_length个标记，这样可以确保输入序列的长度不会超过模型的最大处理能力，同时保留了最相关的上下文信息。

        labels=[-100]*len(prompt_ids)+answer_ids #-100是特殊的标签值，表示这些位置的标记在计算损失时应该被忽略，这样模型在训练过程中只会关注答案部分的标记，而不会受到提示部分的影响，从而更有效地进行监督学习微调。
        if len(labels)>self.max_length:
            labels=labels[-self.max_length:]

        return {
            'input_ids': torch.tensor(input_ids,dtype=torch.long),
            'labels': torch.tensor(labels,dtype=torch.long),
        }

dataset=SFTDataset('./data/alpaca_data_cleaned.json',tokenizer,256)


def collate_fn(batch,pad_id=tokenizer.eos_token_id,label_pad_id=-100):  #用于将__getitem__获得的一个样本整合成一个批次的数据
    max_len=max(x['input_ids'].numel() for x in batch)  #batch包含了一个批次的样本数据
    input_ids=[]
    labels=[]
    for x in batch:
        ids=x['input_ids']
        lab=x['labels']
        pad_len=max_len-len(ids) #计算每个批次中需要填充的长度，确保每一批次的长度相等
        input_ids.append(torch.cat([ids,torch.full((pad_len,),pad_id,dtype=torch.long)]))
        labels.append(torch.cat([lab,torch.full((pad_len,),label_pad_id,dtype=torch.long)]))

    return {'input_ids': torch.stack(input_ids), 'labels': torch.stack(labels)} #将ids和lab堆叠成一个批次的数据，返回一个字典，包含输入ID和标签的批次数据，这些数据将用于模型的训练过程。

dataloader=DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=collate_fn)

args=ModelConfig()

swanlab.init(
    project='my-awesome-project',
    experiment='llm-sft-demo',
    tags=['sft','transformer'],
    config={
        'epochs': 4,
        'batch_size': 4,
        'learning_rate': 3e-4,
        'model_config':args.__dict__
    }
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

        loss=outputs.last_loss.mean() #模型输出的last_loss是一个张量，包含了每个位置的损失值
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
