TEMPLATES={
    'qwen2':("<|im_start|>system\n{system}\n<|im_end|>\n"
        "<|im_start|>user\n{user}\n<|im_end|>\n"
        "<|im_start|>assistant\n"),
    'chatglm2': "###User:\n{instruction}\n{input}\n\n###Assistant:\n",

    
}


def build_prompt(instruction:str,input:str,template:str,system:str='你是一个乐于助人的助手。'):
    if template=='qwen2':
        if input:
            user_content=instruction+'\n'+input
        else:
            user_content=instruction
        prompt=TEMPLATES[template].format(system=system,user=user_content)
        return prompt

    elif template=='chatglm2':
        prompt=TEMPLATES[template].format(instruction=instruction,input=input)
        return prompt

    else:
        raise ValueError(f'Unsupported template: {template}')
    