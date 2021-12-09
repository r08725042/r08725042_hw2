
# coding: utf-8

# In[ ]:


import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from preprocessing_data import *


# In[ ]:


with open('data/train.json') as f:
    data = json.load(f)


# In[ ]:


train_data = []
max_context = 0
max_anwser = 0
for title in data['data']:
    for paragraphs in title['paragraphs']:
        context = paragraphs['context']
        if len(context) > max_context:
            max_context = len(context)
        for temp_qas in paragraphs['qas']:
            temp_data = {'context':context, 'qas':temp_qas}
            if len(temp_qas['question']) > max_anwser:
                max_anwser = len(temp_qas['question'])
                
            train_data.append(temp_data)


# In[ ]:


from transformers import BertTokenizer

pretrained_bert = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(pretrained_bert,do_lower_case=True)


# In[ ]:
print('create train dataset')

train_dataset = create_Bert_dataset(process_bert_samples(tokenizer, train_data))


# In[ ]:


train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = 8,
    shuffle = True,
    collate_fn = lambda x: BertDataset.collate_fn(train_dataset, x)
)


# In[ ]:


from transformers import BertModel
bert_based = BertModel.from_pretrained('bert-base-chinese')


# In[ ]:


class Bert_QA(nn.Module):
    def __init__(self, bert_based):
        super().__init__()
        
        self.bert = bert_based
        self.dropout = nn.Dropout(0.5)
        self.classification = nn.Linear(768,1024)
        self.classification2 = nn.Linear(1024,2)
        self.start = nn.Linear(768,1024)
        self.start2 = nn.Linear(1024,1)
        self.end = nn.Linear(768,1024)
        self.end2 = nn.Linear(1024,1)
        
    def forward(self,inputs, segment, mask):
        output_hidden = self.bert(input_ids=inputs, token_type_ids=segment, attention_mask=mask)
        CLS_hidden = output_hidden[0][:,0,:]
        CLS_hidden = self.dropout(CLS_hidden)
        CLS_output = self.classification(CLS_hidden)
        CLS_output = self.classification2(CLS_output)
        
        START_hidden = END_hidden = output_hidden[0]
        START_hidden = self.dropout(START_hidden)
        START_output = self.start(START_hidden)
        START_output = self.start2(START_output)
        START_output = START_output.squeeze(2)
        END_hidden = self.dropout(END_hidden)
        END_output = self.end(END_hidden)
        END_output = self.end2(END_output)
        END_output = END_output.squeeze(2)
        
        return CLS_output, START_output, END_output


# In[ ]:


bert = Bert_QA(bert_based).cuda()


# In[ ]:


import torch.optim as optim
pos_weight = torch.tensor(0.4447869)
bce_loss = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
start_loss = nn.CrossEntropyLoss(ignore_index = 0)
end_loss = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(bert.parameters(), lr=0.00001, weight_decay=0.0001)


# In[ ]:
print('create dev dataset')

with open('data/dev.json') as f:
    dev = json.load(f)
dev_data = []
max_context = 0
max_anwser = 0
for title in dev['data']:
    for paragraphs in title['paragraphs']:
        context = paragraphs['context']
        if len(context) > max_context:
            max_context = len(context)
        for temp_qas in paragraphs['qas']:
            temp_data = {'context':context, 'qas':temp_qas}
            if len(temp_qas['question']) > max_anwser:
                max_anwser = len(temp_qas['question'])
                
            dev_data.append(temp_data)
dev_dataset = create_Bert_dataset(process_bert_samples(tokenizer, dev_data))


# In[ ]:


dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 8,
    shuffle = False,
    collate_fn = lambda x: BertDataset.collate_fn(dev_dataset, x)
)


# In[ ]:


def eval_bert(model, dev_loader):
    total_bloss=0
    total_sloss=0
    total_eloss=0
    with torch.no_grad():
        model.eval()
        for dev_data in dev_loader:
            inputs = dev_data['inputs']
            segment = dev_data['segment']
            mask = dev_data['mask']
            label = (torch.tensor(dev_data['answerable'])).cuda()
            start_label = (torch.tensor(dev_data['answers_start'])).cuda()
            end_label = (torch.tensor(dev_data['answers_end'])).cuda()
            
            
            CLS_output, START_output, END_output = model(inputs.cuda(), segment.cuda(), mask.cuda())
            
            bloss = bce_loss(CLS_output, label.float())
            sloss = start_loss(START_output, start_label)
            eloss = end_loss(END_output, end_label)
            total_bloss = total_bloss + bloss.item()
            total_sloss = total_sloss + sloss.item()            
            total_eloss = total_eloss + eloss.item()
    
    return total_bloss/len(dev_loader), total_sloss/len(dev_loader), total_eloss/len(dev_loader)


# In[ ]:
print('start training...')

for epoch in range(3):
    count = 0
    print('now epoch : ' , epoch+1)
    bert.train()
    for data in train_loader:
        count = count + 1
        context_len = [len(c) for c in data['context'][1:-1]]
        inputs = data['inputs']
        segment = data['segment']
        mask = data['mask']
        label = (torch.tensor(data['answerable'])).cuda()
        start_label = (torch.tensor(data['answers_start'])).cuda()
        end_label = (torch.tensor(data['answers_end'])).cuda()
        
        
        CLS_output, START_output, END_output = bert(inputs.cuda(), segment.cuda(), mask.cuda())
        optimizer.zero_grad()


        bloss = bce_loss(CLS_output, label.float())
        bloss.backward(retain_graph=True)

        
        
        sloss = start_loss(START_output, start_label)
        sloss.backward(retain_graph=True)
        
        
        eloss = end_loss(END_output, end_label)
        eloss.backward()
        optimizer.step()
        if count%800 ==0:
            print('bloss : ', bloss)
            print('sloss : ', sloss)
            print('eloss : ', eloss)
            print('--------------------------')

    eval_bloss, eval_sloss, eval_eloss = eval_bert(bert, dev_loader)
    print('eval_bloss : ', eval_bloss)
    print('eval_sloss : ', eval_sloss)
    print('eval_eloss : ', eval_eloss)
    print('--------------------------')
    key_name = 'a_bert_weight_' + str(epoch+1) + '.pt'
    torch.save(bert.state_dict(), key_name) 


