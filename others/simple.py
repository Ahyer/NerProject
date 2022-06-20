# -*- encoding : utf-8 -*-
'''
@author : sito
@date : 2022-02-25
@description:
Trying to build model (Bert+BiLSTM+CRF) to solve the problem of Ner,
With low level of code and the persistute of transformers, torch, pytorch-crf
Next Step is to stronger the Training Dataset and text the real data.
'''
import torch
import torch.nn as nn
from transformers import BertModel, AdamW, BertTokenizer
from torchcrf import CRF
import os


class Model(nn.Module):

    def __init__(self, tag_num, max_length):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        config = self.bert.my_config
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2, batch_first=True)
        self.crf = CRF(tag_num)
        self.fc = nn.Linear(config.hidden_size, tag_num)

    def forward(self, x, y):
        with torch.no_grad():
            bert_output = \
            self.bert(input_ids=x.input_ids, attention_mask=x.attention_mask, token_type_ids=x.token_type_ids)[0]
        lstm_output, _ = self.lstm(bert_output)  # (1,30,768)
        fc_output = self.fc(lstm_output)  # (1,30,7)
        # fc_output -> (seq_length, batch_size, n tags) y -> (seq_length, batch_size)
        loss = self.crf(fc_output, y)
        tag = self.crf.decode(fc_output)
        return loss, tag


if __name__ == '__main__':
    # parameters
    epoches = 50
    max_length = 30

    # data preprocess
    x = ["我 和 小 明 今 天 去 了 北 京".split(), "普 京 在  昨 天 进 攻 了 乌 克 拉 ， 造 成 了 大 量 人 员 的 伤 亡".split()]
    y = ["O O B-PER I-PER O O O O B-LOC I-LOC".split(),
         "B-PER I-PER O O O O O O B-LOC I-LOC I-LOC O O O O O O O O O O O".split()]

    tag_to_ix = {"B-PER": 0, "I-PER": 1, "O": 2, "[CLS]": 3, "[SEP]": 4, "B-LOC": 5, "I-LOC": 6}

    labels = []
    for label in y:
        r = [tag_to_ix[x] for x in label]
        if len(r) < max_length:
            r += [tag_to_ix['O']] * (max_length - len(r))
        labels.append(r)
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # input_ids,attention_mask,token_type_ids
    print(x[0])
    tokenizer_result = tokenizer.encode_plus(x[0], return_token_type_ids=True, return_attention_mask=True,
                                             return_tensors='pt',
                                             padding='max_length', max_length=max_length)
    # training
    model = Model(len(tag_to_ix), max_length)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    model.train()
    for i in range(epoches):
        loss, _ = model(tokenizer_result, torch.tensor(labels[0]).unsqueeze(dim=0))
        loss = abs(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f'loss : {loss}')
    # evaluating
    model.eval()
    with torch.no_grad():
        _, tag = model(tokenizer_result, torch.tensor(labels[0]).unsqueeze(dim=0))
        print(f' ori tag: {labels[0]} \n predict tag : {tag}')
    # save model
    # torch.save(model.state_dict(),f'model.pt')
