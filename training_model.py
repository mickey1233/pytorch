import os
import sys
import random

import pickle
import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self,input_num,final_output_num):
        super(MLP,self).__init__()
        
        self.layer = []
        while True:
            self.output_num = input_num
            if self.output_num//2 > final_output_num: 
                self.output_num //= 2
                self.layer.append(nn.Linear(input_num,self.output_num))
                self.layer.append(nn.ReLU())
            else: 
                self.layer.append(nn.Linear(input_num,final_output_num))
                break
            input_num //= 2
        self.layer = nn.Sequential(*self.layer)
        
    def forward(self,x):
        y = self.layer(x)
        return y
def GetDev():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
       
if __name__ == "__main__":
    x1 = [1.0 if (id%2) == 0 else 0.0 for id in range(64)]
    x2 = [1.0 if (id%2) == 0 else 0.0 for id in range(64)]
    x = torch.tensor([x1,x2],dtype=torch.float32)
    print(x.shape)
    
    mlp = MLP(64,10)
    y = mlp(x)
    for x,z in enumerate(y):
        print(x,z)
    
        
    

   
   
   
    
        
        
        
        

        
        