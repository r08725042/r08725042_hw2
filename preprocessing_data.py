
# coding: utf-8

# In[ ]:
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn

def get_tokens_range(tokenizer,sample):
    ranges = []
    char_start = sample['qas']['answers'][0]['answer_start']
    char_end = char_start + len(sample['qas']['answers'][0]['text'])
    token_start = 1 + len(tokenizer.tokenize(sample['context'][:char_start]))
    answer_len = len(tokenizer.tokenize(sample['context'][char_start:char_end]))
    token_end = token_start + answer_len
    ranges = [token_start, token_end]
    return ranges

def process_bert_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        question = (tokenizer.encode(sample['qas']['question'])[1:])
        context = tokenizer.encode(sample['context'])
        if len(context) + len(question) > 512:
            if len(question) > 30:
                question = question[:29] + [question[-1]]
            context = (context[:512-len(question)-1] + [context[-1]])
        processed = {
            'context': context,
            'qid': sample['qas']['id'],
            'question': question,
            'inputs' : context + question
        }

        if 'answers' in sample['qas']:
            processed['answerable'] = [1,0]
            processed['answers_range'] = [0,0]
            if sample['qas']['answerable']:
                processed['answerable'] = [0,1]
                answer_range = get_tokens_range(tokenizer, sample)
                processed['answers_range'] = answer_range

            
                if answer_range[1] > len(context)-1 :
                    continue
            
        processeds.append(processed)
    return processeds

def dev_process_bert_samples(tokenizer, samples):
    processeds = []
    for sample in tqdm(samples):
        question = (tokenizer.encode(sample['qas']['question'])[1:])
        context = tokenizer.encode(sample['context'])
        if len(context) + len(question) > 512:
            if len(question) > 30:
                question = question[:29] + [question[-1]]
            context = (context[:512-len(question)-1] + [context[-1]])
        processed = {
            'context': context,
            'qid': sample['qas']['id'],
            'question': question,
            'inputs' : context + question
        }

        if 'answers' in sample['qas']:
            processed['answerable'] = [1,0]
            processed['answers_range'] = [0,0]
            if sample['qas']['answerable']:
                processed['answerable'] = [0,1]
                answer_range = get_tokens_range(tokenizer, sample)
                processed['answers_range'] = answer_range

#                 dev data can't delete
#                 if answer_range[1] > len(context)-1 :
#                     continue 
            
        processeds.append(processed)
    return processeds


def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds


# In[ ]:


class BertDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance =  {
            'context':sample['context'],
            'qid': sample['qid'],
            'question': sample['question'],
            'inputs': sample['inputs'],
            'segment': [0] * len(sample['context']) + [1] * len(sample['question']),
            'mask': [1] * len(sample['inputs'])
        }
        if 'answerable' in sample:
            instance['answerable'] = sample['answerable']
            instance['answers_start'] = sample['answers_range'][0]
            instance['answers_end'] = sample['answers_range'][1]
            
            
        return instance
             

    def collate_fn(self, samples):
        batch = {}
        for key in ['qid', 'context', 'question', 'answerable', 'answers_start', 'answers_end']:
            if any(key not in sample for sample in samples):
                continue
            batch[key] = [sample[key] for sample in samples]

        for key in ['inputs', 'segment', 'mask']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, 0
            )
            batch[key] = torch.tensor(padded)

        return batch


# In[ ]:


def create_Bert_dataset(samples):
    dataset = BertDataset(samples)
    return dataset

