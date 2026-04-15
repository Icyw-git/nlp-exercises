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
    return ''

if __name__ == '__main__':
    example={
        'messages':[
            {'role':'system','content':'You are a helpful assistant.'},
            {'role':'user','content':'What is the capital of France?'},
            {'role':'assistant','content':'The capital of France is Paris.'}
        ]
    }

    prompt=build_prompt(example)
    print(prompt)