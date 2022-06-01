from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
import torch
import warnings
import os
import json
import sys
import re
warnings.filterwarnings('ignore')

def collect_data(path,original_value,result_value,a,b,c,d,e,f):
    with open(path,'r',encoding='utf-8') as file:
        s = json.load(file)
        # 组织学分型, 癌结节, 两侧切缘是否有癌浸润, pCRM, 脉管, 神经 -> a,b,c,d,e,f
        try:
            for i,k in enumerate(s):
                if k=='originalValue':
                    original_value.append(s['originalValue'])
                if k=='resultValue' and s['resultValue']!='':
                    result_value.append(s['resultValue'])
                if k=='classify':
                    classify_data = s[k]
                    a.append(classify_data['组织学分型']) if "组织学分型" in classify_data else a.append(" ")
                    b.append(classify_data['癌结节']) if "癌结节" in classify_data else b.append(" ")
                    c.append(classify_data['两侧切缘是否有癌浸润']) if "两侧切缘是否有癌浸润" in classify_data else c.append(" ")
                    d.append(classify_data['pCRM']) if "pCRM" in classify_data else d.append(" ")
                    e.append(classify_data['脉管']) if "脉管" in classify_data else e.append(" ")
                    f.append(classify_data['神经']) if "神经" in classify_data else f.append(" ")
        except Exception:
            print(f'Errors occus at path : {path}, key : "{k}", with reasons : {sys.exc_info()}')
    return original_value,result_value,a,b,c,d,e,f

def fun4Word(data):
    output = ''
    for i in data:
        word = ''
        label = ''
        word_label = re.split(r'(\[[^\]]+\]/aj_lcjl|\[[^\]]+\]/aj_hzjl|\[[^\]]+\]/lbj_z|\[[^\]]+\]/lbj_y|\[[^\]]+\]/lbj_fz|\[[^\]]+\]/mlh1|\[[^\]]+\]/msh2|\[[^\]]+\]/msh6|\[[^\]]+\]/pms2|\[[^\]]+\]/ki67|\[[^\]]+\]/p53)',i)
        for f in word_label:
            if 'lbj_y' in f:
                word_index = f[1:-7]
                if len(word_index)>1:
                    label_index = "B_lbjy "+(len(word_index)-2)*'M_lbjy '+"E_lbjy "
                else:
                    label_index = "W_lbjy "
                word += word_index
                label += label_index
            elif 'lbj_z' in f:
                word_index = f[1:-7]
                if len(word_index) > 1:
                    label_index = "B_lbjz " + (len(word_index) - 2)* 'M_lbjz '+ "E_lbjz "
                else:
                    label_index = "W_lbjz "
                word += word_index
                label += label_index
            elif 'lbj_fz' in f:
                word_index = f[1:-8]
                if len(word_index) > 1:
                    label_index = "B_lbjfz " + (len(word_index) - 2)*'M_lbjfz ' + "E_lbjfz "
                else:
                    label_index = "W_lbjfz "
                word += word_index
                label += label_index
            elif 'aj_lcjl' in f:
                word_index = f[1:-9]
                if 'cm' in word_index:
                    if len(word_index) > 3:
                        label_index = "B_ajl " + (len(word_index) - 4) * 'M_ajl ' + "E_ajl "+"O "*2
                    else:
                        label_index = "W_ajl " +"O "*2
                elif 'c' in word_index:
                    if len(word_index) > 2:
                        label_index = "B_ajl " + (len(word_index) - 2) * 'M_ajl ' + "E_ajl " +'O '
                    else:
                        label_index = "W_ajl " +'O '
                else:
                    if len(word_index) > 1:
                        label_index = "B_ajl " + (len(word_index) - 2) * 'M_ajl ' + "E_ajl "
                    else:
                        label_index = "W_ajl "
                word += word_index
                label += label_index
            elif 'aj_hzjl' in f:
                word_index = f[1:-9]
                if 'cm' in word_index:
                    if len(word_index) > 3:
                        label_index = "B_ajh " + (len(word_index) - 4) * 'M_ajh ' + "E_ajh " + "O " * 2
                    else:
                        label_index = "W_ajh " + "O " * 2
                elif 'c' in word_index:
                    if len(word_index) > 2:
                        label_index = "B_ajh " + (len(word_index) - 2) * 'M_ajh ' + "E_ajh " + 'O '
                    else:
                        label_index = "W_ajh " + 'O '
                else:
                    if len(word_index) > 1:
                        label_index = "B_ajh " + (len(word_index) - 2) * 'M_ajh ' + "E_ajh "
                    else:
                        label_index = "W_ajh "
                word += word_index
                label += label_index
            elif 'mlh1' in f:
                word_index = f[1:-6]
                if len(word_index) > 1:
                    label_index = "B_mlh1 " + (len(word_index) - 2) * 'M_mlh1 ' + "E_mlh1 "
                else:
                    label_index = "W_mlh1 "
                word += word_index
                label += label_index
            elif 'msh2' in f:
                word_index = f[1:-6]
                if len(word_index) > 1:
                    label_index = "B_msh2 " + (len(word_index) - 2) * 'M_msh2 ' + "E_msh2 "
                else:
                    label_index = "W_msh2 "
                word += word_index
                label += label_index
            elif 'msh6' in f:
                word_index = f[1:-6]
                if len(word_index) > 1:
                    label_index = "B_msh6 " + (len(word_index) - 2) * 'M_msh6 ' + "E_msh6 "
                else:
                    label_index = "W_msh6 "
                word += word_index
                label += label_index
            elif 'pms2' in f:
                word_index = f[1:-6]
                if len(word_index) > 1:
                    label_index = "B_pms2 " + (len(word_index) - 2) * 'M_pms2 ' + "E_pms2 "
                else:
                    label_index = "W_pms2 "
                word += word_index
                label += label_index
            elif 'ki67' in f:
                word_index = f[1:-6]
                if len(word_index) > 1:
                    label_index = "B_ki67 " + (len(word_index) - 2) * 'M_ki67 ' + "E_ki67 "
                else:
                    label_index = "W_ki67 "
                word += word_index
                label += label_index
            elif 'p53' in f:
                # word_index = f[1:-6]
                word_index = f[1:-5]                           #2020-08-19修改，原值在上一行
                if len(word_index) > 1:
                    label_index = "B_p53 " + (len(word_index) - 2) * 'M_p53 ' + "E_p53 "
                else:
                    label_index = "W_p53 "
                word += word_index
                label += label_index
            else:
                word += f
                label +=len(f)*"O "
        if word !='':
            output += word + ' //' + label + '\n'
    return output

def label_process(data):
    word = ''
    label = ''
    word_label = re.split(
        r'(\[[^\]]+\]/aj_lcjl|\[[^\]]+\]/aj_hzjl|\[[^\]]+\]/lbj_z|\[[^\]]+\]/lbj_y|\[[^\]]+\]/lbj_fz|\[[^\]]+\]/mlh1|\[[^\]]+\]/msh2|\[[^\]]+\]/msh6|\[[^\]]+\]/pms2|\[[^\]]+\]/ki67|\[[^\]]+\]/p53)',data)
    for f in word_label:
        if 'lbj_y' in f:
            word_index = f[1:-7]
            if len(word_index) > 1:
                label_index = "B_lbjy " + (len(word_index) - 2) * 'M_lbjy ' + "E_lbjy "
            else:
                label_index = "W_lbjy "
            word += word_index
            label += label_index
        elif 'lbj_z' in f:
            word_index = f[1:-7]
            if len(word_index) > 1:
                label_index = "B_lbjz " + (len(word_index) - 2) * 'M_lbjz ' + "E_lbjz "
            else:
                label_index = "W_lbjz "
            word += word_index
            label += label_index
        elif 'lbj_fz' in f:
            word_index = f[1:-8]
            if len(word_index) > 1:
                label_index = "B_lbjfz " + (len(word_index) - 2) * 'M_lbjfz ' + "E_lbjfz "
            else:
                label_index = "W_lbjfz "
            word += word_index
            label += label_index
        elif 'aj_lcjl' in f:
            word_index = f[1:-9]
            if 'cm' in word_index:
                if len(word_index) > 3:
                    label_index = "B_ajl " + (len(word_index) - 4) * 'M_ajl ' + "E_ajl " + "O " * 2
                else:
                    label_index = "W_ajl " + "O " * 2
            elif 'c' in word_index:
                if len(word_index) > 2:
                    label_index = "B_ajl " + (len(word_index) - 2) * 'M_ajl ' + "E_ajl " + 'O '
                else:
                    label_index = "W_ajl " + 'O '
            else:
                if len(word_index) > 1:
                    label_index = "B_ajl " + (len(word_index) - 2) * 'M_ajl ' + "E_ajl "
                else:
                    label_index = "W_ajl "
            word += word_index
            label += label_index
        elif 'aj_hzjl' in f:
            word_index = f[1:-9]
            if 'cm' in word_index:
                if len(word_index) > 3:
                    label_index = "B_ajh " + (len(word_index) - 4) * 'M_ajh ' + "E_ajh " + "O " * 2
                else:
                    label_index = "W_ajh " + "O " * 2
            elif 'c' in word_index:
                if len(word_index) > 2:
                    label_index = "B_ajh " + (len(word_index) - 2) * 'M_ajh ' + "E_ajh " + 'O '
                else:
                    label_index = "W_ajh " + 'O '
            else:
                if len(word_index) > 1:
                    label_index = "B_ajh " + (len(word_index) - 2) * 'M_ajh ' + "E_ajh "
                else:
                    label_index = "W_ajh "
            word += word_index
            label += label_index
        elif 'mlh1' in f:
            word_index = f[1:-6]
            if len(word_index) > 1:
                label_index = "B_mlh1 " + (len(word_index) - 2) * 'M_mlh1 ' + "E_mlh1 "
            else:
                label_index = "W_mlh1 "
            word += word_index
            label += label_index
        elif 'msh2' in f:
            word_index = f[1:-6]
            if len(word_index) > 1:
                label_index = "B_msh2 " + (len(word_index) - 2) * 'M_msh2 ' + "E_msh2 "
            else:
                label_index = "W_msh2 "
            word += word_index
            label += label_index
        elif 'msh6' in f:
            word_index = f[1:-6]
            if len(word_index) > 1:
                label_index = "B_msh6 " + (len(word_index) - 2) * 'M_msh6 ' + "E_msh6 "
            else:
                label_index = "W_msh6 "
            word += word_index
            label += label_index
        elif 'pms2' in f:
            word_index = f[1:-6]
            if len(word_index) > 1:
                label_index = "B_pms2 " + (len(word_index) - 2) * 'M_pms2 ' + "E_pms2 "
            else:
                label_index = "W_pms2 "
            word += word_index
            label += label_index
        elif 'ki67' in f:
            word_index = f[1:-6]
            if len(word_index) > 1:
                label_index = "B_ki67 " + (len(word_index) - 2) * 'M_ki67 ' + "E_ki67 "
            else:
                label_index = "W_ki67 "
            word += word_index
            label += label_index
        elif 'p53' in f:
            # word_index = f[1:-6]
            word_index = f[1:-5]  # 2020-08-19修改，原值在上一行
            if len(word_index) > 1:
                label_index = "B_p53 " + (len(word_index) - 2) * 'M_p53 ' + "E_p53 "
            else:
                label_index = "W_p53 "
            word += word_index
            label += label_index
        else:
            word += f
            label += len(f) * "O "
    return word,label

def my_collate(data):
    inputs, labels = [],[]
    for i,dat in enumerate(data):
        (input,label) = dat
        inputs.append(input)
        labels.append(label)
    return torch.tensor(inputs),torch.tensor(labels)

class MyDataSet(Dataset):

    def __init__(self,max_length = 512):
        # parameters
        labels = ['B_lbjy', 'M_lbjy', 'E_lbjy', 'W_lbjy', 'B_lbjz', 'M_lbjz', 'E_lbjz', 'W_lbjz', 'B_lbjfz', 'M_lbjfz',
                  'E_lbjfz', 'W_lbjfz',
                  'B_ajl', 'M_ajl', 'E_ajl', 'W_ajl', 'B_ajh', 'M_ajh', 'E_ajh', 'W_ajh', 'B_mlh1', 'M_mlh1', 'E_mlh1',
                  'W_mlh1', 'B_msh2', 'M_msh2', 'E_msh2', 'W_msh2',
                  'B_msh6', 'M_msh6', 'E_msh6', 'W_msh6', 'B_pms2', 'M_pms2', 'E_pms2', 'W_pms2', 'B_ki67', 'M_ki67',
                  'E_ki67', 'W_ki67', 'B_p53', 'M_p53', 'E_p53', 'W_p53', 'O']
        self.tag_num = len(labels)
        original_value, result_value, a, b, c, d, e, f = [], [], [], [], [], [], [], []
        count = 0
        root = 'data'
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        # collect data
        root_path = os.listdir(root)
        for path in root_path:
            father_path = os.path.join(root, path)
            child_paths = os.listdir(os.path.join(root, path))
            for child_path in child_paths:
                count += 1
                original_value, result_value, a, b, c, d, e, f = collect_data(os.path.join(father_path, child_path),
                                                                              original_value, result_value, a, b, c, d,
                                                                              e, f)
                if result_value=='': print(f'result_value null: count :{count}, path:{os.path.join(father_path, child_path)}')
        print(f'Data Collection Info : original_value : {len(original_value)} result_value : {len(result_value)} '
              f'a : {len(a)} b : {len(b)} c :{len(c)} d : {len(d)} e : {len(e)} f : {len(f)} final count : {count}')
        ### ner data process ###
        # tokenize data and encoding labels
        tokenized_data = []
        encoded_labels = []
        for i,sentence in enumerate(result_value):
            word, label = label_process(sentence)
            # word 预处理， 对于大于max_length的部分阶段
            if len(word)>max_length:
                word = word[:max_length]
            # TODO 添加句子分割方法，将段句分为每段为512的长度
            # 截断大于512字数的句段,小于512的进行填充
            s = tokenizer.encode_plus(word,return_token_type_ids=True,return_attention_mask=True,return_tensors='pt',
                                             padding='max_length',max_length=max_length)
            tokenized_data.append(s)
            #add label information, 并对label进行编码
            label = label.strip().split(' ')
            # 截断超出512的部分，填充小于512的部分，并编码label
            if len(label)>max_length:
                label = label[:max_length]
            if len(label)<max_length:
                label += ['O'] * (max_length-len(label))
            # encoding label
            l = {k: v for v, k in enumerate(labels)}
            encoded_label = [l[k] for k in label]
            encoded_labels.append(encoded_label)
            if s.input_ids.shape[1]>max_length or s.attention_mask.shape[1]>max_length or s.token_type_ids.shape[1]>max_length:
                print(f'len data:{s.input_ids.shape} {s.attention_mask.shape} {s.token_type_ids.shape} len label:{len(encoded_label)}')
        self.data = tokenized_data
        self.label = encoded_labels
        # TODO add classification data process
        # ...

    def __getitem__(self, index):
        return self.data[index],self.label[index]

    def __len__(self):
        return len(self.data)

###############################
if __name__ == '__main__':
    dataset = MyDataSet()
    token_count = 0
    data_loader = DataLoader(dataset=dataset,shuffle=False,batch_size=10,collate_fn=my_collate)
    for i,data in enumerate(data_loader):
        inputs,labels = data
        print(f'inputs_size:{inputs.shape}\t labels_size:{labels.shape}')
        token_count +=1
    print(f'token_count:{token_count}')

# output = fun4Word(result_value)
# with open('output.txt','w',encoding='utf-8') as file:
#     file.write(output)
#     file.close()
