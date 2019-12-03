import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Input,Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD


batch_size = 128
dataset_path = './data_set/'

filename = os.listdir(dataset_path)
train_percentage = 95
img_width = 100
img_height = 100
n_calss = len(os.listdir(dataset_path))


def get_datalist(datapath,train_percentage=90):
    train_datapath = []
    train_label = []
    test_datapath = []
    test_label = []

    filename = os.listdir(datapath)
    for i,path in enumerate(filename):
       dataname  = os.listdir(datapath+path)
       print(path,len(dataname))
       for file in dataname:
           chance = np.random.randint(100)
           if chance < train_percentage:
               train_datapath.append(datapath + path+ '/' + file)
               train_label.append(i)
           else:
               test_datapath.append(datapath + path + '/' + file)
               test_label.append(i)
    print('train data:',len(train_datapath))
    print('test data:',len(test_datapath))
    return [train_datapath,train_label,test_datapath,test_label]


def get_random_batch(train_datapath,train_label,batchsize,n_calss,img_width,img_height):
    
    train_data = np.zeros([batchsize,img_width,img_height,3])
    train_data =  train_data.astype(np.uint8)
    train_label_onehot = np.zeros([batchsize,n_calss])
    
    l = len(train_datapath)
    i = 0
    for _ in range(batchsize):
        image_index = random.randrange(l)
        img = cv2.imread(train_datapath[image_index])
        train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))
        train_label_onehot[i,int(train_label[image_index])] = 1
#        print(i,image_index,train_datapath[image_index])
        i += 1
    return train_data,train_label_onehot

# 加载文件路径
train_datapath,train_label,test_datapath,test_label = get_datalist(dataset_path,train_percentage)
train_data,train_label_onehot = get_random_batch(train_datapath,
                                                 train_label,
                                                 batch_size,
                                                 n_calss,
                                                 img_width,
                                                 img_height)

input_shape = (img_width, img_height, 3)
input_tensor=Input(shape=input_shape)
x = Conv2D(32, (7, 7), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

 # Block 2
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Classification block
# x = Flatten(name='flatten')(x)
x = GlobalAveragePooling2D(name = 'Average_pooling')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(64, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictions')(x)

model = Model(inputs= input_tensor, outputs = x)
optimizer = SGD()
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer , loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
quit()
for i in range(5000):
    train_data,train_label_onehot = get_random_batch(train_datapath,
                                                 train_label,
                                                 batch_size,
                                                 n_calss,
                                                 img_width,
                                                 img_height)
    train_data = train_data.astype('float')
    train_loss, train_accuracy = model.train_on_batch(train_data, train_label_onehot)
    print('iteration:',i,'loss:',train_loss,'accuracy:',train_accuracy)
#    


    