import torch
import os
from torch.utils.data import Dataset
from transformers import BertTokenizer


class NerDataSet(Dataset):
    def __init__(self, mode="train"):
        self.src, self.tag = self.get_file(os.getcwd() + "\Data\\" + mode + ".txt")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tag[idx]

    def get_file(self, path):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        list_src = []
        list_tag = []
        _src = []
        _tag = []
        with open(path, mode='r', encoding='utf-8') as f:
            for line in f:
                line = line.strip("\n")
                if (len(line) == 0):
                    list_src.append(_src[:])
                    list_tag.append(_tag[:])
                    _src.clear()
                    _tag.clear()
                else:
                    array = line.split(' ')
                    text, label = array[0], array[1],
                    s = tokenizer.encode_plus(text, return_token_type_ids=True, return_attention_mask=True,
                                              return_tensors='pt',
                                              padding='max_length', max_length=512).to('cuda')
                    _src.append(s)
                    _tag.append(label)
        if len(_src) != 0:
            list_src.append(_src)
            list_tag.append(_tag)
        return list_src, list_tag


def my_collate(data):
    tag_to_ix = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "X": 7, "[CLS]": 8,
                 "[SEP]": 9}
    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    print(len(data))
    inputs, labels = [], []
    for i, dat in enumerate(data):
        (input, label) = dat
        inputs.append(input)
        labels.append(label)


if __name__ == '__main__':
    dataset = NerDataSet(mode="test")
    print(len(dataset))
    ui = []
    ui.append(dataset[0])
    ui.append(dataset[1])
    ui.append(dataset[2])
    print(len(ui))
    my_collate(ui)
