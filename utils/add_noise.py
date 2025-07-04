import numpy as np
import random
from PIL import Image,ImageFilter
import torch
from torchvision import transforms


class RandomChannelErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.erase = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=0)

    def __call__(self, input):
        c_dim = input.size(1)
        for c in range(c_dim):
            input[:,c,:,:] = self.erase(input[:,c,:,:])
        return input

class RandomFrameErasing(object):
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.erase = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=0)

    def __call__(self, input):
        f_dim = input.size(0)
        for f in range(f_dim):
            input[f,:,:,:] = self.erase(input[f,:,:,:])
        return input

if __name__=="__main__":
    t = RandomFrameErasing(1)
    # t = RandomFrameErasing(1)
    pred = torch.ones([4, 3, 8, 8], dtype=torch.float32)
    pred = t(pred)
    print('----------- month1 c0~c2 -----------')
    print(pred[0, 0])
    print(pred[0, 1])
    print(pred[0, 2])
    print('----------- month2 c0~c2 -----------')
    print(pred[1, 0])
    print(pred[1, 1])
    print(pred[1, 2])
    print('----------- month3 c0~c2 -----------')
    print(pred[2, 0])
    print(pred[2, 1])
    print(pred[2, 2])
    print('----------- month4 c0~c2 -----------')
    print(pred[3, 0])
    print(pred[3, 1])
    print(pred[3, 2])
    # print('----------- c0 month1~month4 -----------')
    # print(pred[0, 0])
    # print(pred[1, 0])
    # print(pred[2, 0])
    # print(pred[3, 0])
