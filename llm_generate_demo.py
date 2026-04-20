import torch
from transformers import AutoTokenizer


from llm_LLaMA2 import ModelConfig,Transformer


@torch.inference_mode()
def generate_eos(inputs,check_point_path:str,tokenizer,temperature:float,top_k:int,max_length:int=256):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args=ModelConfig()
    model=Transformer(args)
    model.load_state_dict(torch.load(check_point_path,map_location='cpu'))
    model.to(device)
    model.eval()

    prompt=f"###User:\n{inputs}\n\n###Assistant:\n"
    input_ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)['input_ids'].to(device)



    if input_ids.numel() > max_length:
        input_ids = input_ids[:,-max_length:]

    for _ in range(max_length):
        outputs=model(input_ids)
        logits=outputs.logits[:,-1,:] #获取最后一个位置的logits，表示下一个标记的预测结果

        if temperature==0.0:
            idx=torch.argmax(logits,dim=-1,keepdim=True)

        else:
            next_token_logits = logits / temperature
            v,_=torch.topk(next_token_logits,min(top_k,next_token_logits.size(-1)))
            next_token_logits=next_token_logits.masked_fill(next_token_logits<v[:,[-1]], float('-inf'))
            probs=torch.softmax(next_token_logits,dim=-1)
            idx=torch.multinomial(probs,num_samples=1)

        input_ids=torch.cat([input_ids,idx],dim=-1)

        if idx.item()==tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids,skip_special_tokens=True)





if __name__=='__main__':
    tokenizer = AutoTokenizer.from_pretrained('tokenizer')

    generate_eos('今天天气怎么样？','check_point_path',tokenizer,temperature=0.7,top_k=50)









    