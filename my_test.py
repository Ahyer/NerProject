import torch

if __name__ == '__main__':
    a = torch.tensor([1, 2, 3]).cuda()
    b = torch.tensor([1, 23, 3]).cuda()
    cal = a == b
    count=torch.where(cal>0)[0].shape[0]
    list=[1,2,3,4,5,6,9]
    print(torch.mean(torch.LongTensor( list)))
    print(count)
