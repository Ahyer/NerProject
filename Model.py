import torch
import torch.nn as nn
from transformers import BertModel, AdamW, BertTokenizer
from torchcrf import CRF
import torch.optim as optim
import logging
import time
from torch.utils.data import Dataset, DataLoader
import numpy as np
from NerDataSet import NerDataSet as nds, my_collate

logger = logging.getLogger('training log')
logger.setLevel(logging.INFO)


class Model(nn.Module):
    def __init__(self, tag_num):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        config = self.bert.config
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=config.hidden_size,
                            hidden_size=config.hidden_size // 2, batch_first=True)
        self.crf = CRF(tag_num)
        self.fc = nn.Linear(config.hidden_size, tag_num)

    def forward(self, x, y):
        with torch.no_grad():
            bert_output = \
                self.bert(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                          token_type_ids=x['token_type_ids'])[0]
        lstm_output, _ = self.lstm(bert_output)  # (1,30,768)
        fc_output = self.fc(lstm_output)  # (1,30,7)
        # fc_output -> (seq_length, batch_size, n tags) y -> (seq_length, batch_size)
        loss = None
        if y != None:
            loss = self.crf(fc_output, y)
        tag = self.crf.decode(fc_output)
        return loss, tag


def train_loop(dataloader, model, optimizer, scheduler, e):
    epoch_end_loss = 0
    model.train()
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        inputs, labels = data
        loss, _ = model(inputs, labels)
        loss = abs(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_end_loss = loss
        if i % 10 == 0:
            print(f'>>> epoch {e + 1} <<< step {i} : loss : {loss}')
    print(
        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e + 1} training loss : {epoch_end_loss}')


def test_loop(dataloader, model, e):
    model.eval()
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e + 1} Start Evaluation!')
    step_end_accuracy = []
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs, labels = data
            _, tag = model(inputs, labels)
            tag = torch.tensor(tag).to(device).T
            # tag = np.array(tag).T
            # calculate the precision
            for i, (pre_y, real_y) in enumerate(zip(tag, labels)):
                assert pre_y.shape[0] == real_y.shape[0] == max_length, \
                    f'length not match pre_y.shape[0]:{pre_y.shape[0]} real_y.shape[0]:{real_y.shape[0]}  max_length:{max_length}'
                sum = pre_y.shape[0]
                # real_y_numpy = real_y.cpu().numpy()
                cal = pre_y == real_y
                count = torch.where(cal > 0)[0].shape[0]
                # count = np.where(cal > 0)[0].size
                accu = count / sum
                step_end_accuracy.append(accu)

    epoch_end_accuracy = np.mean(step_end_accuracy)
    print(
        f'{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} epoch {e + 1} evaluation accuracy : {epoch_end_accuracy}')
    # save model
    torch.save(model.state_dict(), f'model_p_{epoch_end_accuracy}.pt')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    epoches = 100
    max_length = 50
    lr = 0.0001
    batch_size = 128
    model = Model(tag_num=10).to(device)
    train_loader = DataLoader(dataset=nds("train"), batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    dev_loader = DataLoader(dataset=nds('dev'), batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    test_loader = DataLoader(dataset=nds('test'), batch_size=batch_size, shuffle=True, collate_fn=my_collate)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    for e in range(epoches):
        print(f"Epoch {e + 1}\n-------------------------------")
        train_loop(train_loader, model=model, optimizer=optimizer, scheduler=scheduler, e=e)
        test_loop(dev_loader, model=model, e=e)
