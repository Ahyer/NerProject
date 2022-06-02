# -*- encoding : utf-8 -*-
'''
@author : sito
@date : 2022-02-25
@description:
Trying to build model (Bert+BiLSTM+CRF) to solve the problem of Ner,
With low level of code and the persistute of transformers, torch, pytorch-crf
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from test import MyDataSet
from torch.utils.data import DataLoader
from transformers import BertModel
from torchcrf import CRF
import time
import warnings
import logging
import sys
warnings.filterwarnings('ignore')
# log configuration
logger = logging.getLogger('training log')
logger.setLevel(logging.INFO)
# stream handler
rf_handler = logging.StreamHandler(sys.stderr)
rf_handler.setLevel(logging.INFO)
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
# file handler
f_handler = logging.FileHandler('output/training.log')
f_handler.setLevel(logging.INFO)
f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
logger.addHandler(rf_handler)
logger.addHandler(f_handler)

def my_collate(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids, attention_mask, token_type_ids, labels = [],[],[],[]
    for i,dat in enumerate(data):
        (input,label) = dat
        input_ids.append(input.input_ids.cpu().squeeze().detach().numpy().tolist())
        attention_mask.append(input.attention_mask.cpu().squeeze().detach().numpy().tolist())
        token_type_ids.append(input.token_type_ids.cpu().squeeze().detach().numpy().tolist())
        labels.append(label)
    return {'input_ids': torch.tensor(input_ids).to(device), 'attention_mask':torch.tensor(attention_mask).to(device),
            'token_type_ids':torch.tensor(token_type_ids).to(device)}, torch.tensor(labels).to(device)

class Model(nn.Module):

    def __init__(self,tag_num):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        config = self.bert.config
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size, hidden_size=config.hidden_size//2, batch_first=True)
        self.crf = CRF(tag_num)
        self.fc = nn.Linear(config.hidden_size,tag_num)

    def forward(self,x,y):
        with torch.no_grad():
            bert_output = self.bert(input_ids=x['input_ids'],attention_mask=x['attention_mask'],token_type_ids=x['token_type_ids'])[0]
        lstm_output, _ = self.lstm(bert_output) # (batch_size,seq_len,hidden_size)
        fc_output = self.fc(lstm_output) # (batch_size,seq_len,tag_num)
        loss = self.crf(fc_output,y) # y (batch_size,seq_len)
        tag = self.crf.decode(fc_output) # (tag_num)
        return loss,tag

if __name__ == '__main__':
    # parameters
    epoches = 50
    max_length = 512
    batch_size = 64 # 32
    lr = 0.0001 # 5e-4  0.0001
    # data preprocess
    dataset = MyDataSet(max_length)
    tag_num = dataset.tag_num
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=my_collate)
    # training
    print(f'>>> Training Start!')
    model = Model(tag_num).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)
    for e in range(epoches):
        # training
        epoch_end_loss = 0
        model.train()
        for i,data in enumerate(data_loader):
            optimizer.zero_grad()
            inputs, labels = data
            loss,_ = model(inputs,labels)
            loss = abs(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_end_loss = loss
            if i%10==0:
                print(f'>>> epoch {e} <<< step {i} : loss : {loss}')
        print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e} training loss : {epoch_end_loss}')
        # evaluating
        if e%10==0 and e!=0:
            model.eval()
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e} Start Evaluation!')
            step_end_accuracy = []
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    inputs, labels = data
                    _, tag = model(inputs,labels)
                    tag = np.array(tag).T
                    # calculate the precision
                    for i,(pre_y,real_y) in enumerate(zip(tag,labels)):
                        assert pre_y.shape[0]==real_y.shape[0]==max_length, \
                            f'length not match pre_y.shape[0]:{pre_y.shape[0]} real_y.shape[0]:{real_y.shape[0]}  max_length:{max_length}'
                        sum = pre_y.shape[0]
                        real_y_numpy= real_y.cpu().numpy()
                        cal = pre_y==real_y_numpy
                        count = np.where(cal>0)[0].size
                        accu = count/sum
                        step_end_accuracy.append(accu)

            epoch_end_accuracy = np.mean(step_end_accuracy)
            print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e} evaluation accuracy : {epoch_end_accuracy}')
            # save model
            torch.save(model.state_dict(),f'model_p_{epoch_end_accuracy}.pt')
