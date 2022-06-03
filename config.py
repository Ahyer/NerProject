import os


class Config:
    def __init__(self):
        self.pretrain_model_path = os.path.join(os.getcwd(), 'bert-base-chinese')
