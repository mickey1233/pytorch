import glob
import os
import sys
import pathlib
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
class GetTrainingData(Dataset):
    def __init__(self,path):
        self.training_data_list = []
        self.file_path = glob.glob(os.path.join(path,'*','*.dgp'))
        for i in self.file_path:
            data = i.split(os.sep)
            label = data[-2]
            image_pixel = []
            
            with open(i,'r',encoding='utf-8') as f:
                file = f.readlines()
                image_pixel = [0.0 if y == '0' else 1.0 for j in file for y in j if y == '0' or y == '1' ]
                self.training_data_list.append([image_pixel,int(label)])
        
    def __getitem__(self,index1):
        return torch.tensor(self.training_data_list[index1][0], dtype=torch.float32), \
               torch.tensor(self.training_data_list[index1][1], dtype =torch.long)
    
    def __len__(self):
        return len(self.training_data_list)

if __name__ == '__main__':
    folder_path = pathlib.Path(__file__).parent.absolute()
    print(folder_path)
    path = os.path.join(folder_path,'TrainData')
    get_data = GetTrainingData(path)
    
    get_data_loader = DataLoader(dataset=get_data,shuffle=True,batch_size=2)
    for img,label in get_data_loader:
        print(img,label)
        
            
            