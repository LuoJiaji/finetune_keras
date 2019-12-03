import os
import random

dataset_path = './data_set/'


def get_datalist(datapath):
    
    filename = os.listdir(datapath)
    datafile = []
    label = []
    cnt = 0
    print('number of category:', len(filename), filename)
    for i,path in enumerate(filename):
        dataname  = os.listdir(os.path.join(datapath, path))
        print(path,':',len(dataname))
        curr_path = []
        curr_label = []
        for file in dataname:
#            datafile.append(os.path.join(datapath,path,file))
            curr_path.append(datapath+path+'/'+file)
            curr_label.append(i)
        cnt += len(curr_path)
        datafile.append(curr_path)
        label.append(curr_label)
    print('*'*40)
    print('Data Count:',cnt)
    return datafile, label

def data_split_by_number(datafile, label):
    l = len(datafile)
    split_num = [100, 100, 100, 100, 100]
    test_data_path = []
    test_label = []
    for i in range(l):
        num = len(datafile[i])
        curr_test_data_path = []
        curr_test_label = []
        ind_set = set()
        while len(ind_set) != split_num[i]:
            ind_set.add(random.randrange(num))

        for ind in ind_set:
            curr_test_data_path.append(datafile[i][ind])
            curr_test_label.append(i)
        
        for p in curr_test_data_path:
            datafile[i].remove(p)
            label[i].remove(i)
        test_data_path.append(curr_test_data_path) 
        test_label.append(curr_test_label)
    train_data_path = datafile
    train_label = label
    
    return train_data_path, train_label, test_data_path, test_label

datapath, label = get_datalist(dataset_path)
train_data_path, train_label, test_data_path, test_label = data_split_by_number(datapath, label)

for i in range(len(datapath)):
    print('test number:', len(test_data_path[i]), len(test_label[i]), 
          'train number:', len(train_data_path[i]), len(train_label[i]))
