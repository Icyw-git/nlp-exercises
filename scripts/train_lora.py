#通过transformers库完成llm训练和开发

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from torch.utils.data import DataLoader
import torch
from transformers import TrainingArguments,Trainer
import swanlab
from dotenv import load_dotenv

from src.data.sft_dataset import SFTDataset
from src.data.collate import collate_fn

from functools import partial
from src.training.config import load_config
cfg=load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Configs', 'sft_lora.yaml'))



load_dotenv()
api_key=os.getenv("SWANLAB_API_KEY")
swanlab.login(api_key)


from peft import LoraConfig,get_peft_model,TaskType
lora_config=LoraConfig(
    task_type=TaskType.CAUSAL_LM, #指定任务类型
    r=cfg.lora.r, #秩，表示LoRA微调中低秩矩阵的秩，较小的r值可以减少模型参数的数量，从而降低训练和推理的计算成本，但可能会影响模型的性能。选择合适的r值需要根据具体任务和模型进行实验和调整，以找到性能和效率之间的最佳平衡点。
    lora_alpha=cfg.lora.lora_alpha, #alpha值，表示LoRA微调中低秩矩阵的缩放因子，较大的lora_alpha值可以增加模型的表达能力，从而提高性能，但也可能增加训练和推理的计算成本。选择合适的lora_alpha值需要根据具体任务和模型进行实验和调整，以找到性能和效率之间的最佳平衡点。
    lora_dropout=cfg.lora.lora_dropout, #dropout率，表示在LoRA微调过程中应用的dropout率，较高的dropout率可以增加模型的鲁棒性和泛化能力，但也可能导致训练过程中的不稳定性。选择合适的lora_dropout值需要根据具体任务和模型进行实验和调整，以找到性能和稳定性之间的最佳平衡点。
    target_modules=cfg.lora.target_modules, #指定要应用LoRA微调的目标模块，这些模块通常是Transformer模型中的线性层，例如查询、键、值和输出投影层。通过指定这些模块，LoRA微调将只针对这些特定的层进行参数调整，从而提高训练效率和性能。选择合适的target_modules需要根据具体任务和模型结构进行实验和调整，以找到最佳的微调效果。
    bias='none'
)


from transformers import AutoConfig
model_path=cfg.model.model_path #模型下载路径
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

data_path=cfg.data.train_path




with open(data_path,'r',encoding='utf-8') as f:
    text=json.load(f)

train_data=text[:int(len(text)*0.9)]
val_data=text[int(len(text)*0.9):]

train_dataset=SFTDataset(train_data,tokenizer,max_length=cfg.data.max_length,template=cfg.data.template,from_list=True)
val_dataset=SFTDataset(val_data,tokenizer,max_length=cfg.data.max_length,template=cfg.data.template,from_list=True)


train_dataloader=DataLoader(train_dataset,batch_size=cfg.training.batch_size,shuffle=True,collate_fn=partial(collate_fn,pad_id=tokenizer.eos_token_id))
val_dataloader=DataLoader(val_dataset,batch_size=cfg.training.batch_size,shuffle=False,collate_fn=partial(collate_fn,pad_id=tokenizer.eos_token_id))

for batch in train_dataloader:
    print(batch)
    break

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=get_peft_model(model,lora_config) #使用peft库中的get_peft_model函数将原始模型转换为支持LoRA微调的模型，lora_config参数包含了LoRA微调的配置，例如任务类型、秩、alpha值、dropout率等，这些配置将指导模型在训练过程中如何应用LoRA微调技术，从而提高训练效率和性能。
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.training.learning_rate)# optimizer要在模型转换为LoRA模型之后定义，因为get_peft_model函数会修改模型的参数结构，如果在转换之前定义优化器，可能会导致优化器无法正确地识别和更新模型的参数，从而影响训练过程。因此，建议在调用get_peft_model函数之后再定义优化器，以确保优化器能够正确地识别和更新模型的参数。

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
    per_device_train_batch_size=cfg.training.batch_size,
    num_train_epochs=cfg.training.epochs,
    logging_steps=cfg.training.logging_steps, #每隔多少步记录一次日志，这些日志可以包括训练损失、评估指标等信息，帮助我们监控训练过程中的模型性能和收敛情况。
    save_steps=cfg.training.save_steps, #每隔多少步保存一次模型检查点，这些检查点可以用于后续的模型恢复、评估或部署，确保我们在训练过程中不会丢失重要的模型状态。
    output_dir=cfg.training.output_dir,
    fp16=cfg.training.fp16, #启用混合精度训练，这可以加速训练过程并减少显存使用，特别是在使用GPU进行训练时，fp16可以提高计算效率，同时保持模型的性能。
    report_to=['swanlab'],
    eval_strategy='steps', #评估策略，指定在训练过程中何时进行评估，这里设置为'steps'表示每隔一定的训练步骤进行一次评估，评估的频率可以通过eval_steps参数来控制，这样可以帮助我们监控模型在验证集上的性能，并及时调整训练过程中的超参数或模型结构，以获得更好的性能。
    eval_steps=cfg.training.eval_steps, #每隔500步进行一次评估操作，使用的是默认的评估指标，这些指标可以帮助我们监控模型在验证集上的性能，并及时调整训练过程中的超参数或模型结构，以获得更好的性能。
)


trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=partial(collate_fn, pad_id=tokenizer.eos_token_id),
    optimizers=(optimizer,None),
    tokenizer=tokenizer,
)

#Todo:lora训练流程优化
trainer.train()
