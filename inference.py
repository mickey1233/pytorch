import pathlib
import os
import time

import get_training_data as gtd 
from training_model import MLP
import training_model as tm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

if __name__ == "__main__":
    ##GET DATA##
    folder_path = pathlib.Path(__file__).parent.absolute()
    path = os.path.join(folder_path,'TrainData')
    data = gtd.GetTrainingData(path)
    data_loader = DataLoader(dataset=data,batch_size=4)
    
    
    folder_path = pathlib.Path(__file__).parent.absolute()
    model = tm.MLP(64,10)
    model.load_state_dict(torch.load(os.path.join(folder_path,'model','train_model'),map_location=tm.GetDev()))
    
    loss = nn.CrossEntropyLoss
    count = 0
    acc = 0
    start_time = time.time()
    
    for (train_data,label) in data_loader:
        #print(train_data.shape)
        #print((train_data.squeeze(0)).shape)
        
        train_list = []
        
        actual_data = F.one_hot(label,num_classes=10).type(torch.float32)
        
        
        model.eval()
        with torch.no_grad():
            for i,label in zip(train_data,label):
                count+=1
                train_model = model(i)
                value,label1 = torch.max(train_model,0)
                
                print('第{}筆data'.format(count),'預測結果 =',train_model,'分類結果 =',label1,'實際分類 =',label)
                if label1 == label:
                    acc+=1
    end_time = time.time()
    print('預測正確數量 = ', acc,'總數量 =', count,'準確度 = ', acc/count)
    print('預測總花費時間 =', end_time-start_time)
    '''   
        for i in train_model:
            train_list.append(i.value)
        end1_time = time.time()
        print('第{}筆data'.format(count),'預測結果 =',train_list,'最大值 =',max(train_list),'分類結果 = ',train_list.index(max(train_list)),'實際分類 =',label,'loss =',train_loss.value,'預測花費時間 =', end1_time-start1_time)
        if train_list.index(max(train_list)) == label:
            acc+=1
        count+=1
    end_time = time.time()
    count-=1
    print('預測正確數量 = ', acc,'總數量 =', count,'準確度 = ', acc/count)
    print('預測總花費時間 =', end_time-start_time)
    '''