#通过transformers库完成llm训练和开发

import os
import json
from torch.utils.data import Dataset,DataLoader
import torch
from transformers import TrainingArguments,Trainer
import swanlab
from dotenv import load_dotenv

load_dotenv()
api_key=os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)


from peft import LoraConfig,get_peft_model,TaskType
lora_config=LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=['q_proj','k_proj'],
    bias='none'
)


from transformers import AutoConfig
model_path='./models/Qwen2.5-1.5B' #模型下载路径
config=AutoConfig.from_pretrained(model_path) #加载模型配置，AutoConfig是transformers库中的一个类，用于加载预训练模型的配置文件，from_pretrained方法接受一个模型路径作为参数，返回一个包含模型配置的对象，这些配置参数包括模型的层数、隐藏维度、注意力头数等，这些参数将用于定义和训练Transformer模型。
print(config)

from transformers import AutoModelForCausalLM

model=AutoModelForCausalLM.from_pretrained(model_path) #加载模型配置和权重

# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(name)


from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained(model_path) #加载模型分词器

print(tokenizer.eos_token_id)

data_path='./data/alpaca_data_cleaned.json'




with open(data_path,'r',encoding='utf-8') as f:
    text=json.load(f)

train_data=text[:int(len(text)*0.9)]
val_data=text[int(len(text)*0.9):]

class SFTDataset(Dataset):
    def __init__(self,data,tokenizer,max_length):
        super().__init__()
        self.data=data
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        example=self.data[idx]
        instruction=example['instruction']
        input=example.get('input','')
        output=example.get('output','')

        if input:
                user_content=instruction+'\n'+input
        else:
                user_content=instruction

        prompt = (
                "<|im_start|>system\n你是一个乐于助人的助手。<|im_end|>\n"
                "<|im_start|>user\n" + user_content + "<|im_end|>\n"
                "<|im_start|>assistant\n"
        )

        prompt_ids=self.tokenizer.encode(prompt,add_special_tokens=False)
        answer=self.tokenizer.encode(output,add_special_tokens=False)

        if tokenizer.eos_token_id is not None:
            answer.append(tokenizer.eos_token_id)

        input_ids=prompt_ids+answer

        labels=[-100]*len(prompt_ids)+answer
        if len(labels)>self.max_length:
            labels=labels[-self.max_length:]

        if len(input_ids)>self.max_length:
            input_ids=input_ids[-self.max_length:]

        return {
            'input_ids': torch.tensor(input_ids,dtype=torch.long),
            'labels': torch.tensor(labels,dtype=torch.long)
        }


train_dataset=SFTDataset(train_data,tokenizer,max_length=512)
val_dataset=SFTDataset(val_data,tokenizer,max_length=512)



def collate_fn(batch,pad_id=tokenizer.eos_token_id,label_pad_id=-100):
    max_len=max(x['input_ids'].numel() for x in batch)
    input_ids=[]
    labels=[]

    for x in batch:
        ids=x['input_ids']
        labs=x['labels']
        pad_len=max_len-len(ids)
        ids=torch.cat([ids,torch.full((pad_len,),pad_id,dtype=torch.long)])
        labs=torch.cat([labs,torch.full((pad_len,),label_pad_id,dtype=torch.long)])
        input_ids.append(ids)
        labels.append(labs)
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }




train_dataloader=DataLoader(train_dataset,batch_size=4,shuffle=True,collate_fn=collate_fn)
val_dataloader=DataLoader(val_dataset,batch_size=4,shuffle=False,collate_fn=collate_fn)


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()



training_args=TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    output_dir='./models/qwen2.5-lora',
    fp16=True,
    report_to=['swanlab']
)


trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    optimizers=(optimizer,None),
    tokenizer=tokenizer,
)



trainer.train()
