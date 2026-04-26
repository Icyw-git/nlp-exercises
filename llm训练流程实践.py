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
    target_modules=['q_proj','k_proj','v_proj','o_proj'],
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

        input_ids=input_ids[:self.max_length] #这里采用的是前截断的方式，如果输入文本超过了最大长度，就保留输入文本的最后部分，这样可以确保模型能够看到输入文本的结尾部分，这对于生成回答可能更有帮助，因为输入文本的结尾部分通常包含了最相关的信息。
        labels=labels[:self.max_length]
        assert len(input_ids)==len(labels)

        return {
            'input_ids': torch.tensor(input_ids,dtype=torch.long),
            'labels': torch.tensor(labels,dtype=torch.long)
        }


train_dataset=SFTDataset(train_data,tokenizer,max_length=512)
val_dataset=SFTDataset(val_data,tokenizer,max_length=512)



def collate_fn(batch,pad_id=tokenizer.eos_token_id,label_pad_id=-100):
    max_len=max(x['input_ids'].numel() for x in batch) #统计每个批次中的最长输入长度，以便进行动态填充
    input_ids=[]
    labels=[]

    for x in batch:
        ids=x['input_ids']
        labs=x['labels']
        pad_len=max_len-len(ids)
        ids=torch.cat([ids,torch.full((pad_len,),pad_id,dtype=torch.long)]) #将输入ID进行填充，使用pad_id来填充输入ID，使得每个批次中的输入ID长度相同，这样可以方便地将它们堆叠成一个张量进行模型训练。
        labs=torch.cat([labs,torch.full((pad_len,),label_pad_id,dtype=torch.long)])
        input_ids.append(ids)
        labels.append(labs)
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }




train_dataloader=DataLoader(train_dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
val_dataloader=DataLoader(val_dataset,batch_size=2,shuffle=False,collate_fn=collate_fn)

for batch in train_dataloader:
    print(batch)
    break

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=get_peft_model(model,lora_config) #使用peft库中的get_peft_model函数将原始模型转换为支持LoRA微调的模型，lora_config参数包含了LoRA微调的配置，例如任务类型、秩、alpha值、dropout率等，这些配置将指导模型在训练过程中如何应用LoRA微调技术，从而提高训练效率和性能。
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)# optimizer要在模型转换为LoRA模型之后定义，因为get_peft_model函数会修改模型的参数结构，如果在转换之前定义优化器，可能会导致优化器无法正确地识别和更新模型的参数，从而影响训练过程。因此，建议在调用get_peft_model函数之后再定义优化器，以确保优化器能够正确地识别和更新模型的参数。

model.train()
model.print_trainable_parameters()

batch = next(iter(train_dataloader))
for k in batch:
    batch[k] = batch[k].to(model.device)
model.train()
out = model(input_ids=batch["input_ids"], labels=batch["labels"])
print("loss:", out.loss)
out.loss.backward()

for n, p in model.named_parameters():
    if p.requires_grad:
        grad_norm = p.grad.norm().item() if p.grad is not None else None
        print(f"{n}: grad_norm={grad_norm}")



training_args=TrainingArguments(
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=50, #每隔多少步记录一次日志，这些日志可以包括训练损失、评估指标等信息，帮助我们监控训练过程中的模型性能和收敛情况。
    save_steps=500,
    output_dir='./models/qwen2.5-lora',
    fp16=True, #启用混合精度训练，这可以加速训练过程并减少显存使用，特别是在使用GPU进行训练时，fp16可以提高计算效率，同时保持模型的性能。
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
