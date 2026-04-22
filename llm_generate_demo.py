from typing import Optional

import torch
from transformers import AutoTokenizer
import torch.nn.functional as F



from llm_LLaMA2 import ModelConfig,Transformer


@torch.inference_mode() #推理模式,禁用梯度计算。
def generate_eos(inputs:str, check_point_path:str, tokenizer, temperature:float, top_k:Optional[int]=None, top_p: Optional[float]=None, repetition_penality:float=1.0,max_length:int=256):  #check_point_path是模型权重的路径
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args=ModelConfig() #初始化模型配置参数
    model=Transformer(args)
    checkpoint=torch.load(check_point_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict']) #加载模型权重，map_location='cpu'表示将模型权重加载到CPU上，这样可以确保在没有GPU的环境中也能正确加载模型权重，避免出现设备不兼容的问题。
    model.to(device)
    model.eval()

    prompt=f"###User:\n{inputs}\n\n###Assistant:\n"
    input_ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(device) #add_special_tokens=False表示在编码时不添加特殊标记，例如开始标记、结束标记等，这样可以确保输入ID只包含用户输入的内容，而不包含任何额外的特殊标记，这对于生成任务来说是有意义的，因为我们希望模型根据用户输入生成回答，而不是受到特殊标记的干扰。

    #返回的是一个二维张量，形状为[1, sequence_length]，其中1表示批次大小，sequence_length表示输入文本的长度。通过选择第一个元素（input_ids[0]），我们得到一个一维张量，表示输入文本的分词结果，这样可以方便地进行后续的生成操作。


    if input_ids.numel() > max_length:
        input_ids = input_ids[:,-max_length:] #截断

    for _ in range(max_length):
        outputs=model(input_ids)
        logits=outputs.logits[:,-1,:] #输出的Logits是[batch_size,1,vocab_size]获取最后一个位置的logits，表示下一个标记的预测结果，维度为(batch_size, vocab_size)，其中batch_size是输入的批次大小，vocab_size是词汇表的大小。

        if repetition_penality>1.0:  #repetition_penality是一个浮点数，表示重复惩罚的程度，如果大于1.0，则对已经生成过的标记进行惩罚，降低它们被再次生成的概率，从而鼓励模型生成更多样化的文本。通过检查repetition_penality是否大于1.0，我们可以决定是否需要对已经生成过的标记进行惩罚处理。
            seen=input_ids[0]
            logits[:,seen]=logits[:,seen]/repetition_penality

        if temperature==0.0:
            idx=torch.argmax(logits,dim=-1,keepdim=True) #这里的维度是(batch_size, 1)，表示每个输入序列的下一个标记的索引，keepdim=True表示保持维度不变，这样在后续的操作中可以方便地与输入ID进行拼接。

        else:
            next_token_logits = logits / temperature #使用温度对logits进行缩放

            if top_k is not None and top_k>0:
                v,_=torch.topk(next_token_logits,min(top_k,next_token_logits.size(-1))) #避免top_k大于词汇表大小的情况
                next_token_logits=next_token_logits.masked_fill(next_token_logits<v[:,[-1]], float('-inf')) #将不在top_k范围内的标记的logits设置为负无穷，这样在softmax计算概率时，这些标记的概率将接近于零，从而确保只从top_k的标记中进行采样。

            if top_p is not None and 0<top_p<1.0:
                sorted_logits,sorted_idxs=torch.sort(next_token_logits,dim=-1,descending=True) #排序函数，对logits进行排序，dim=-1表示在最后一个维度上进行排序，descending=True表示降序排序，这样可以得到从高到低的logits值和对应的索引。
                sorted_probs=F.softmax(sorted_logits,dim=-1)
                cumulative_probs=sorted_probs.cumsum(dim=-1) #计算累计和，cumsum函数用于计算累积和，dim=-1表示在最后一个维度上进行计算，这样可以得到每个标记的累计概率，从而确定哪些标记的累计概率超过了top_p的阈值。


                mask=cumulative_probs>top_p #生成掩码，mask是一个布尔张量，表示哪些标记的累计概率超过了top_p的阈值，这些标记将被掩码处理，即在后续的操作中被设置为负无穷，从而确保只从累计概率不超过top_p的标记中进行采样。
                mask[...,0]=False #这里[...,0]的意思是[:,0]，表示对每个输入序列的第一个标记不进行掩码处理，确保至少保留一个标记在top_p范围内，这样可以避免出现没有可选标记的情况。



                sorted_logits=sorted_logits.masked_fill(mask,float('-inf'))

                next_token_logits=torch.zeros_like(sorted_logits).scatter(dim=-1,index=sorted_idxs,src=sorted_logits) #将排序后的logits重新放回原来的位置，scatter函数用于将排序后的logits按照原来的索引位置重新放回到next_token_logits中，dim=-1表示在最后一个维度上进行操作，index=sorted_idxs表示按照排序后的索引位置进行放置，src=sorted_logits表示使用排序后的logits作为源数据，这样可以得到经过top_p过滤后的logits。

            logits=F.softmax(next_token_logits,dim=-1)
            idx=torch.multinomial(logits,num_samples=1)





        input_ids=torch.cat([input_ids,idx],dim=-1) #将得到的下一个标记进行拼接

        eos_id=tokenizer.eos_token_id
        if eos_id is not None and idx.item()==eos_id:
            break


    #只返回包含assistant回答的部分，去掉prompt部分
    out= tokenizer.decode(input_ids,skip_special_tokens=True) #返回生成的文本，skip_special_tokens=True表示在解码时跳过特殊标记，例如<|im_start|>、<|im_end|>等. 注意返回的类型是List类型
    out=out.split('###Assistant:\n',1)[-1].strip() #从生成的文本中提取出助手的回答部分，split('###Assistant:\n',1)将生成的文本按照'###Assistant:\n'进行分割，得到一个列表，[-1]表示获取分割后的最后一个元素，即助手的回答部分，strip()方法用于去除回答部分的前后空白字符，确保输出的文本干净整洁。
    return out




if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained('tokenizer')

    generate_eos('今天天气怎么样？','./checkpoint/checkpoint_s11000_e3.pth',tokenizer,temperature=0.7,top_k=50,repetition_penality=1.2,max_length=256)









    