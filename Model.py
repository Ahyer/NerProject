import torch
import torch.nn as nn
from transformers import BertModel, AdamW, BertTokenizer
from torchcrf import CRF


class Model(nn.Module):
    def __init__(self, tag_num, max_length):
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
                self.bert(input_ids=x.input_ids, attention_mask=x.attention_mask, token_type_ids=x.token_type_ids)[0]
        lstm_output, _ = self.lstm(bert_output)  # (1,30,768)
        fc_output = self.fc(lstm_output)  # (1,30,7)
        # fc_output -> (seq_length, batch_size, n tags) y -> (seq_length, batch_size)
        loss = self.crf(fc_output, y)
        tag = self.crf.decode(fc_output)
        return loss, tag


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
