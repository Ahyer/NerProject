import os


class Config:
    def __init__(self):
        self.pretrain_model_path = os.path.join(os.getcwd(), 'bert-base-chinese')
        self.tt = ["O", "B-ENTI", "I-ENTI", "B-ORG", "I-ORG", "B-ADD", "I-ADD", "B-CON", "I-CON", "B-LAW", "I-LAW",
                   "B-PHONE", "I-PHONE", "B-ID", "I-ID", "B-BANK", "I-BANK", "B-EMAIL", "I-EMAIL", "X", "[CLS]",
                   "[SEP]"]
        self.max_length = 50
