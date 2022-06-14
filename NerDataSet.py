import torch
from torch.utils.data import Dataset, DataLoader

import os
from torch.utils.data import Dataset
from transformers import BertTokenizer

from config import Config

model_path = os.path.join(os.getcwd(), 'bert-base-chinese')


class NerDataSet(Dataset):
    def __init__(self, mode="train"):
        self.max_length = Config().max_length
        # self.tt = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        self.tt = ["O", "B-ENTI", "I-ENTI", "B-ORG", "I-ORG", "B-ADD", "I-ADD", "B-CON", "I-CON", "B-LAW", "I-LAW",
                   "B-PHONE", "I-PHONE", "B-ID", "I-ID", "B-BANK", "I-BANK", "B-EMAIL", "I-EMAIL", "X", "[CLS]",
                   "[SEP]"]
        self.data_path = os.path.join(os.getcwd(), "contract_data", f"{mode}.txt")
        self.src, self.tag, = self.get_file(self.data_path)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tag[idx]

    def get_file(self, path):
        tokenizer = BertTokenizer.from_pretrained(Config().pretrain_model_path)
        list_src = []
        list_tag = []
        _src = []
        _tag = []
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip("\n")
                if (len(line) == 0):
                    __ = tokenizer.encode_plus(_src, return_token_type_ids=True, return_attention_mask=True,
                                               return_tensors='pt', truncation=True,
                                               padding='max_length', max_length=self.max_length).to('cuda')
                    list_src.append(__)
                    dict = {word: index for index, word in enumerate(Config().tt)}
                    list_tag.append([dict[k] for k in _tag])
                    if (len(list_tag[-1]) < self.max_length):
                        list_tag[-1].extend([0 for i in range(self.max_length - len(list_tag[-1]))])
                    else:
                        list_tag[-1] = list_tag[-1][0: self.max_length]
                    _src.clear()
                    _tag.clear()
                else:
                    array = line.split(' ')
                    text, label = array[0], array[1],
                    _src.append(text)
                    _tag.append(label)
        return list_src, list_tag


def my_collate(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids, attention_mask, token_type_ids, labels = [], [], [], []
    for i, dat in enumerate(data):
        (input, label) = dat
        input_ids.append(input.input_ids.cpu().squeeze().detach().numpy().tolist())
        attention_mask.append(input.attention_mask.cpu().squeeze().detach().numpy().tolist())
        token_type_ids.append(input.token_type_ids.cpu().squeeze().detach().numpy().tolist())
        labels.append(label)
        torch.tensor(input_ids).to(device)
        torch.tensor(token_type_ids).to(device)
        torch.tensor(attention_mask).to(device)
    return {'input_ids': torch.tensor(input_ids).to(device), 'attention_mask': torch.tensor(attention_mask).to(device),
            'token_type_ids': torch.tensor(token_type_ids).to(device)}, \
           torch.tensor(labels).to(device)


if __name__ == '__main__':
    dataset = NerDataSet(mode="test")
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=my_collate)
    for i, data in enumerate(data_loader):
        print("年后")
