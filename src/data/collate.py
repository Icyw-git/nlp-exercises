import torch
def collate_fn(batch,pad_id,label_pad_id=-100):
    max_len=max(x['input_ids'].numel() for x in batch)
    input_ids=[]
    label_ids=[]
    for x in batch:
        ids=x['input_ids']
        labs=x['labels']
        pad_len=max_len-len(ids)
        input_ids.append(torch.cat([ids,torch.full((pad_len,),pad_id,dtype=torch.long)]))
        label_ids.append(torch.cat([labs,torch.full((pad_len,),label_pad_id,dtype=torch.long)]))

    return {
        'input_ids':torch.stack(input_ids),
        'labels':torch.stack(label_ids)
    }

