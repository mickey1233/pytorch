import get_training_data as gtd


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import time
import pathlib
import training_model as tm
import os
import sys
import matplotlib.pyplot as plt



if __name__ == "__main__":
    train_total_loss_list = []
    test_total_loss_list = []
    epoch_list = []
    
    folder_path = pathlib.Path(__file__).parent.absolute()
    train_path = os.path.join(folder_path,'TrainData')
    
    dev = tm.GetDev()
    print('dev =',dev)
    train_data = gtd.GetTrainingData(train_path)
    train_data_loader = DataLoader(train_data,batch_size=4)
    
    test_path = os.path.join(folder_path,'TestData')
    test_data = gtd.GetTrainingData(test_path)
    test_data_loader = DataLoader(test_data,batch_size=4)
    
    model = tm.MLP(64,10)
    my_optim = optim.SGD(model.parameters(),lr=0.01)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(100): 
        train_total_loss = 0.0
        test_total_loss = 0.0
        model.train()
        for (train_data,label) in train_data_loader:
            my_optim.zero_grad()
            actual_data = F.one_hot(label,num_classes=10).type(torch.float32)
            train_data = train_data.to(dev)
            train_model =  model(train_data)
            train_loss = loss(train_model,actual_data)
            train_total_loss += train_loss.item()
            train_loss.backward()
            my_optim.step()
        
        model.eval()
        for (test_data,label) in test_data_loader:
            actual_data = F.one_hot(label,num_classes=10).type(torch.float32)
            test_data = test_data.to(dev)
            test_model =  model(test_data)
            test_loss = loss(test_model,actual_data)
            test_total_loss += test_loss.item()
        print(epoch+1,'train_total_loss =',train_total_loss,'test_total_loss =',test_total_loss)
        train_total_loss_list.append(train_total_loss)
        test_total_loss_list.append(test_total_loss)
        epoch_list.append(epoch)
    end_time = time.time()
    print('訓練總花費時間 =', end_time-start_time)
    torch.save(model.state_dict(),os.path.join(folder_path,'model','train_model'))
    
    #slm.save_model('train_model',model.parameters())
    plt.plot(epoch_list,train_total_loss_list,epoch_list,test_total_loss_list)

    plt.title("loss",fontsize=24)
    plt.xlabel("epoch",fontsize=14)
    plt.ylabel("total_loss",fontsize=14)
    plt.savefig('Loss.jpg')
    plt.show()
    