import glob
import os
import sys
import pathlib
import numpy as np
def DataAugmentation(folder_path):
    data_label = 0
    test_path = os.path.join(folder_path,'TestData')
    train_path = os.path.join(folder_path,'TrainData')
    file_path = glob.glob(os.path.join(test_path,'*','*.dgp'))
    for i in file_path:
        #print(i)
        data = i.split(os.sep)
        column_image_pixel = []
        label = data[-2]
        row_image_pixel = []
        
        tmp = []
        if data[-2] != data_label: 
            data_label = data[-2]
        with open(i,'r',encoding='utf-8') as f:
            file = f.readlines()
            for i in file:
                for j in i:
                    if j == '1' or j == '0':
                        tmp.append(int(j))
                    if j == '\n':
                        row_image_pixel.append(tmp)
                        tmp = []
            
            row_index = [i for i in range(len(row_image_pixel)) if 1 not in row_image_pixel[i]]
            column_image_pixel = ((np.array(row_image_pixel)).T).tolist()
            print('column_image_pixel =',column_image_pixel,'row_image_pixel =', row_image_pixel)
            column_index = [i for i in range(len(column_image_pixel)) if 1 not in column_image_pixel[i]]
            if row_index != []:
                index = move_row_bit(row_image_pixel[:],row_index,train_path,label)
            if column_index != []:
                move_column_bit(column_image_pixel[:],column_index,train_path,label,index)

def move_row_bit(data,index,train_path,label):
    column_image_pixel = []
    column_image_pixel = ((np.array(data)).T).tolist()
    column_index = [i for i in range(len(column_image_pixel)) if 1 not in column_image_pixel[i]]
    index1 = 1
    for i in index:
        if i < 4:
            for j in range(1,i+2):
                result_list = []
                result_list.extend(data[j:])
                result_list.extend(data[:j])
                if index1 == 14:
                    break
                writefile(os.path.join(train_path,str(label),"Number{}_{}.dgp".format(str(label),str(index1))),result_list[:])
                index1+=1
                
                if column_index != []:
                    index1 = move_column_bit(result_list[:],column_index,train_path,label,index1)
                 
        elif i >=4:
            for j in range(1,9-i):
                result_list = []
                result_list.extend(data[-j:])
                result_list.extend(data[:-j])
                if index1 == 14:
                    break
                writefile(os.path.join(train_path,str(label),"Number{}_{}.dgp".format(str(label),str(index1))),result_list[:])
                index1+=1
                if column_index != []:
                    index1 = move_column_bit(result_list[:],column_index,train_path,label,index1)            
    return index1
def move_column_bit(data,index,train_path,label,file_index):
    for i in index:
        if i < 4:
            for j in range(1,i+2):
                result_list = []
                result_list.extend(data[j:])
                result_list.extend(data[:j])
                result_list = ((np.array(result_list)).T).tolist()
                if file_index == 14:
                    break
                writefile(os.path.join(train_path,str(label),"Number{}_{}.dgp".format(str(label),str(file_index))),result_list)
                file_index+=1
        elif i >=4:
            for j in range(1,9-i):
                result_list = []
                result_list.extend(data[-j:])
                result_list.extend(data[:-j])
                result_list = ((np.array(result_list)).T).tolist()
                if file_index == 14:
                    break
                writefile(os.path.join(train_path,str(label),"Number{}_{}.dgp".format(str(label),str(file_index))),result_list)
                file_index+=1
    return file_index

def writefile(path,content):
    lines = ''
    with open(path,"w") as f:
        for i in content:
            for j in i:
                lines+=str(j)
                lines+=' '
            lines+='\n'
            
        f.writelines(lines)

if __name__ == "__main__":
    folder_path = pathlib.Path(__file__).parent.absolute()

    path = os.path.join(folder_path,'TestData')
    get_data = DataAugmentation(folder_path)