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