from transformers import AutoTokenizer


def build_prompt(example):
    if 'messages' in example:
        return build_chat_prompt(example['messages'])

    else:
        return build_sft_prompt(example)

def build_chat_prompt(messages):
    prompt=[]
    for m in messages:
        role=m['role']
        content=m['content']
        if role=='system':
            prompt.append(f'<|system|>\n{content}\n')
        elif role=='user':
            prompt.append(f'<|user|>\n{content}\n')
        elif role=='assistant':
            prompt.append(f'<|assistant|>\n{content}\n')

    prompt.append('<|assistant|>\n')
    return "".join(prompt)

def build_sft_prompt(example):
    instruction=example.get('instruction')
    input=example.get('input')

    PROMPT_TEMPLATE=(
        "指令：{instruction}\n输入：{input}\n回答：\n"
    )

    if input:
        prompt=PROMPT_TEMPLATE.format(instruction=instruction, input=input)
    else:
        prompt= f"指令：{instruction}\n回答：\n"

    return prompt




def encode_prompt(example, tokenizer,max_length):
    messages=example['messages']
    prompt=build_prompt(example)
    prompt_ids=tokenizer.encode(prompt, add_special_tokens=False)
    answer=messages[-1]['content'] # 将最后一个消息的内容作为监督学习的答案
    answer_ids=tokenizer.encode(answer, add_special_tokens=False)

    eos_ids=tokenizer.eos_token_id
    if eos_ids is not None:
        answer_ids.append(eos_ids)


    input_ids=prompt_ids+answer_ids

    if len(input_ids)>max_length:
        input_ids=input_ids[-max_length:]

    labels=[-100]*len(prompt_ids)+answer_ids
    labels=labels[-max_length:]

    return input_ids, labels


def collate(batch):
    pass





if __name__ == '__main__':
    example={
        'messages':[
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':'What is the capital of France?'},
            {'role':'assistant','content':'The capital of France is Paris.'}
        ]
    }

    example1={
        'instruction':'What is the capital of France?',
        'input':''
    }

    example2={'instruction':'What is the capital of France?','input':'The capital of France is Paris.'}

    prompt=build_prompt(example)
    print(prompt)

    prompt1=build_prompt(example1)
    print(prompt1)
    prompt2=build_prompt(example2)
    print(prompt2)

    example3={
        'messages':[
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':'What is the capital of France?'},
            {'role':'assistant','content':'The capital of France is Paris.'}
        ]
    }

    tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')
    input_ids, labels=encode_prompt(example3, tokenizer, max_length=512)
    print(input_ids)
    print(labels)

