
# coding: utf-8

# In[ ]:


import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import pickle
from preprocessing_data import *
from argparse import ArgumentParser
from pathlib import Path


# In[ ]:


parser = ArgumentParser()
parser.add_argument('--test_data_path')
parser.add_argument('--output_path')
args = parser.parse_args()


# In[ ]:


from transformers import BertTokenizer

pretrained_bert = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(pretrained_bert)


# In[ ]:


test_path = Path(args.test_data_path)
with open(test_path) as f:
    dev = json.load(f)
o_dev_data = []
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
                
            o_dev_data.append(temp_data)
print('create dataset')
dev_dataset = create_Bert_dataset(dev_process_bert_samples(tokenizer, o_dev_data))


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


dev_loader = DataLoader(
    dataset = dev_dataset,
    batch_size = 4,
    shuffle = False,
    collate_fn = lambda x: BertDataset.collate_fn(dev_dataset, x)
)


# In[ ]:


bert = Bert_QA(bert_based).cuda()


# In[ ]:


bert.load_state_dict(torch.load('bert_weight.pt'))


# In[ ]:

print('now predicting...')
pred_qid = []
pred_answerable = []
pred_start = []
pred_end = []
input_context = []
with torch.no_grad():
    bert.eval()
    for dev_data in dev_loader:
        for each_context in dev_data['context']:
            input_context.append(each_context)
        inputs = dev_data['inputs']
        segment = dev_data['segment']
        mask = dev_data['mask']
        CLS_output, START_output, END_output = bert(inputs.cuda(), segment.cuda(), mask.cuda())
        pred_qid.extend(dev_data['qid'])
        temp_answerable = CLS_output.argmax(1).cpu().tolist()
        pred_answerable.extend(temp_answerable)
        temp_start = START_output.argmax(1).cpu().tolist()
        temp_end = END_output.argmax(1).cpu().tolist()
        pred_start.extend(temp_start)
        pred_end.extend(temp_end)


# In[ ]:


predict_data = {}

for i in range(len(pred_qid)):
    if (pred_answerable[i]):
        if (pred_start[i] < pred_end[i]):
            if (pred_end[i]-pred_start[i]<=30):
                temp_answer = tokenizer.decode(input_context[i][pred_start[i]:pred_end[i]])
                temp_answer = temp_answer.replace(" ", "")
                predict_data[pred_qid[i]] = temp_answer
            elif(pred_end[i]-pred_start[i]<=80):
                temp_answer = tokenizer.decode(input_context[i][pred_start[i]:pred_end[i]])
                temp_answer = temp_answer.replace(" ", "")
                predict_data[pred_qid[i]] = temp_answer[:30]
            else:
                predict_data[pred_qid[i]] = ""
        else:
            predict_data[pred_qid[i]] = ""

                
    else:
        predict_data[pred_qid[i]] = ""


# In[ ]:


output_file = Path(args.output_path)
with open(output_file, 'w')as f:
    json.dump(predict_data, f)

