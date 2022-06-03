from transformers import BertTokenizer

from Model import Model
import torch

from config import Config


def predict(str_input, tokenizer):
    str_list = list(str_input)
    input = tokenizer.encode_plus(str_list, return_token_type_ids=True, return_attention_mask=True,
                                  return_tensors='pt', truncation=True,
                                  padding='max_length', max_length=50).to('cuda')
    input_ids = input.input_ids.cpu().squeeze().detach().numpy().tolist()
    attention_mask = input.attention_mask.cpu().squeeze().detach().numpy().tolist()
    token_type_ids = input.token_type_ids.cpu().squeeze().detach().numpy().tolist()
    dict = {'input_ids': torch.tensor(input_ids).unsqueeze(dim=0).to(device),
            'attention_mask': torch.tensor(attention_mask).unsqueeze(dim=0).to(device),
            'token_type_ids': torch.tensor(token_type_ids).unsqueeze(dim=0).to(device)}
    _, tag = model(dict, None)
    print(tag)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model(tag_num=10).to(device)
    tokenizer = BertTokenizer.from_pretrained(Config().pretrain_model_path)
    # 模型位置
    model.load_state_dict(torch.load('/home/ahyer/code/NerProject/model_p_0.9860051768766179.pt'))
    model.eval()
    while True:
        str_input = input("请输入")
        predict(str_input, tokenizer=tokenizer)
