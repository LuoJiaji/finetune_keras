import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from keras.layers import Lambda
from keras.optimizers import SGD, RMSprop
from keras.layers.normalization import BatchNormalization
# from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix

batch_size = 64
img_width = 224
img_height = 224
# STATE = 'eval'
STATE = 'train'

train_dataset_path = './dataset12/noise/train/'
test_data_path = './dataset12/noise/test/'

filename = os.listdir(train_dataset_path)
n_calss = len(os.listdir(train_dataset_path))


def get_datalist(datapath):
    
    filename = os.listdir(datapath)
    datafile = []
    label = []
    
    print(filename)
    for i,path in enumerate(filename):
        dataname  = os.listdir(os.path.join(datapath,path))
        print(path,':',len(dataname))
        
        for file in dataname:
#            datafile.append(os.path.join(datapath, path, file))
            datafile.append(datapath + path + '/' + file)
            label.append(i)
        
    return datafile,label


# def get_datalist(datapath,train_percentage=90):
#     train_datapath = []
#     train_label = []
#     test_datapath = []
#     test_label = []

#     filename = os.listdir(datapath)
#     for i,path in enumerate(filename):
#        dataname  = os.listdir(datapath+path)
#        print(path,len(dataname))
#        for file in dataname:
#            chance = np.random.randint(100)
#            if chance < train_percentage:
#                train_datapath.append(datapath + path+ '/' + file)
#                train_label.append(i)
#            else:
#                test_datapath.append(datapath + path + '/' + file)
#                test_label.append(i)
#     print('train data:',len(train_datapath))
#     print('test data:',len(test_datapath))
#     return [train_datapath,train_label,test_datapath,test_label]


def get_random_batch(train_datapath,train_label,batchsize,n_calss,img_width,img_height):
    
    train_data = np.zeros([batchsize,img_width,img_height,3])
    # train_data =  train_data.astype(np.uint8)
    train_label_onehot = np.zeros([batchsize,n_calss])
    
    l = len(train_datapath)
    i = 0
    for _ in range(batchsize):
        image_index = random.randrange(l)
        # img = cv2.imread(train_datapath[image_index])
        # train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))

        img = image.load_img(train_datapath[image_index], target_size=(img_height, img_width))
        img = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        img = preprocess_input(img)
        train_data[i,:,:,:]  = img

        train_label_onehot[i,int(train_label[image_index])] = 1
#        print(i,image_index,train_datapath[image_index])
        i += 1
    return train_data,train_label_onehot

my_concat = Lambda(lambda x: K.concatenate([x[0],x[1],x[2],x[3],x[4]],axis=-1))

# 加载文件路径
train_datapath, train_label = get_datalist(train_dataset_path)
test_datalist, test_label = get_datalist(test_data_path) 

# train_data,train_label_onehot = get_random_batch(train_datapath,
#                                                  train_label,
#                                                  batch_size,
#                                                  n_calss,
#                                                  img_width,
#                                                  img_height)

base_model = VGG16()
# base_model.summary()

# interlayer_weights = base_model.get_layer('block2_conv2').get_weights()
# print(interlayer_weights[0].shape)

# interlayer_weights = base_model.get_layer('block5_conv2').get_weights()
# print(interlayer_weights[0].shape)

# VGG16一共23层
layers =  base_model.layers
num_layers = len(base_model.layers)
print('number of layers:',num_layers)

print('layer name:') 
for layers in base_model.layers[:num_layers]:
     layers.trainable = False
     print(layers.name)
    # print(type(layers))
print('*'*50)
print('trainable weights:')
for x in base_model.trainable_weights:
    print(x.name)

# x = GlobalAveragePooling2D(name='average_pooling')(base_model.get_layer('block5_conv3').output)
# x = base_model.get_layer('block5_conv3').output
# x = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN')(x)
# x = GlobalAveragePooling2D(name='average_pooling')(x)
# x = base_model.get_layer('fc2').output

x1 = base_model.get_layer('block1_conv2').output
x1 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN')(x1)
x1 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN1')(x1)
x1 = BatchNormalization(name= 'bn_block1_CNN1')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_CNN_pool1')(x1)
x1 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN2')(x1)
x1 = BatchNormalization(name= 'bn_block1_CNN2')(x1)
x1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_CNN_pool2')(x1)
x1 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN3')(x1)
x1 = BatchNormalization(name= 'bn_block1_CNN3')(x1)
x1 = GlobalAveragePooling2D(name='average_pooling1')(x1)

x2 = base_model.get_layer('block2_conv2').output
x2 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block2_CNN1')(x2)
x2 = BatchNormalization(name= 'bn_block2_CNN1')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_CNN_pool1')(x2)
x2 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block2_CNN2')(x2)
x2 = BatchNormalization(name= 'bn_block2_CNN2')(x2)
x2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_CNN_pool2')(x2)
x2 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block2_CNN3')(x2)
x2 = BatchNormalization(name= 'bn_block2_CNN3')(x2)
x2 = GlobalAveragePooling2D(name='average_pooling2')(x2)

x3 = base_model.get_layer('block3_conv3').output
x3 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block3_CNN1')(x3)
x3 = BatchNormalization(name= 'bn_block3_CNN1')(x3)
x3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_CNN_pool')(x3)
x3 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block3_CNN2')(x3)
x3 = BatchNormalization(name= 'bn_block3_CNN2')(x3)
x3 = GlobalAveragePooling2D(name='average_pooling3')(x3)

x4 = base_model.get_layer('block4_conv3').output
x4 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block4_CNN')(x4)
x4 = BatchNormalization(name= 'bn_block4_CNN')(x4)
x4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_CNN_pool')(x4)
x4 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block4_CNN2')(x4)
x4 = BatchNormalization(name= 'bn_block4_CNN2')(x4)
x4 = GlobalAveragePooling2D(name='average_pooling4')(x4)
# x4 = Flatten()(x4)

x5 = base_model.get_layer('block5_conv3').output
x5 = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block5_CNN')(x5)
x5 = GlobalAveragePooling2D(name='average_pooling5')(x5)

# print(x1.shape, x2.shape, x3.shape)
x = my_concat([x1, x2, x3, x4, x5])
x = Dense(128,activation='relu', name = 'fc_1')(x)
x = Dense(128,activation='relu', name = 'fc_2')(x)
# x = Dropout(0.5)(x)
prediction = Dense(n_calss,activation='softmax',name='fc_output')(x)
# print(x.shape)
              
model = Model(inputs=base_model.input, outputs=prediction)

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# plot_model(model, to_file='./model_visualization/VGG16_highway.png',show_shapes=True)
# model.save('./model/keras/VGG16_fc.h5')
# quit()

# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

                     
#img = load_img('./img/me.jpg',target_size=(224,224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#pre = base_model.predict(x)

# 图像Generator，用来构建输入数据
# train_datagen = ImageDataGenerator(width_shift_range=0,
#                                   height_shift_range=0,
#                                   zoom_range=0,
#                                   horizontal_flip=False)

# train_generator = train_datagen.flow_from_directory('./data_set', 
#                                                    target_size = (224, 224), 
#                                                    batch_size = batch_size)

# image_numbers = train_generator.samples
# class_numbers = train_generator.class_indices
# print('image numbers:',image_numbers)
# print('class',class_numbers)


##model.fit_generator(train_generator,
##                    steps_per_epoch = image_numbers, 
##                    batch_size, epochs = epochs, 
##                    validation_data = validation_generator, 
##                    validation_steps = batch_size)
# 
#
# history = model.fit_generator(train_generator,
#                    steps_per_epoch = 3, 
#                    epochs = epochs)
# log = history.history

print('STATE:',STATE)

if STATE == 'train':
    STEPS = 100000
    log_file = './logs/finetune_VGG16_highway_logs.csv'
    acc_file = './logs/finetune_VGG16_highway_acc.csv'
    f_log = open(log_file,'w')
    f_log.write('iter,loss,train acc'+'\n')
    f_log.flush()
    f_acc = open(acc_file,'w')
    f_acc.write('iter,acc'+'\n')
    f_acc.flush()

    for it in range(STEPS):
        train_data,train_label_onehot = get_random_batch(train_datapath,
                                                        train_label,
                                                        batch_size,
                                                        n_calss,
                                                        img_width,
                                                        img_height)
        # train_data = train_data.astype('float')
        train_loss, train_accuracy = model.train_on_batch(train_data, train_label_onehot)
        
        temp = model.predict(train_data)
    #    quit()

        print('iteration:',it,'loss:',train_loss,'accuracy:',train_accuracy)
        f_log.write(str(it)+','+str(train_loss)+','+str(train_accuracy)+'\n')
        f_log.flush()

        if (it+1) % 1000 == 0 or it + 1 == STEPS:
            pre = []
            l = len(test_datalist)
            for i, path in enumerate(test_datalist):
                if i%100 == 0:
                    print('\r','test:',i,'/',l,end = '')
                
                # img = cv2.imread(path)
                # img  = cv2.resize(img,(img_height,img_width))
                # # print(img.shape)
                # img = np.expand_dims(img,axis = 0)
                # print(img.shape)
                img = image.load_img(path, target_size=(img_width, img_width))
                img = image.img_to_array(img)
                img = np.expand_dims(img, axis=0)
                img = preprocess_input(img)
                pre += [np.argmax(model.predict(img))]
            pre = np.array(pre)    
            test_label = np.array(test_label)
            # print('pre shape:',pre.shape,'test label shape:',test_label.shape)
            acc = np.mean(pre==test_label)
            print('\r','*'*50)
            print('test accuracy:',acc)
            f_acc.write(str(it)+','+str(acc)+'\n')
            f_acc.flush()

    model.save('./models/keras/finetune_VGG16_highway.h5')    
    f_log.close()
    f_acc.close()

elif STATE == 'eval':
    model = load_model('./model/keras/finetune_VGG16_highway.h5')
    result_file = './model/keras/results_divide_VGG16_highway.csv'
    # model.summary()
    # quit()
#    pre = []
#    l = len(test_datalist)
#    for i, path in enumerate(test_datalist[:15]):
#        if i%10 == 0:
#            print('\r','test:',i,'/',l,end = '')
#        
#        # img = cv2.imread(path)
#        # img  = cv2.resize(img,(img_height,img_width))
#        # # print(img.shape)
#        # img = np.expand_dims(img,axis = 0)
#        # print(img.shape)
#        img = image.load_img(path, target_size=(img_width, img_width))
#        img = image.img_to_array(img)
#        img = np.expand_dims(img, axis=0)
#        img = preprocess_input(img)
#        pre += [np.argmax(model.predict(img))]
#    pre = np.array(pre)    
#    test_label = np.array(test_label)
#    # print('pre shape:',pre.shape,'test label shape:',test_label.shape)
#    acc = np.mean(pre == test_label)
#    print('\r','*'*50)
#    print('test accuracy:',acc)
#    
    # 计算混淆矩阵
    # re = confusion_matrix(test_label , pre)
    # print('\n'*2,'-'*10,'confuse matrix','-'*10)
    # print(re)

    # 计算各信号各信噪比下准确率
    Signal_kind = os.listdir(test_data_path)
    Signal_SNR= os.listdir(test_data_path+Signal_kind[0])
    Signal_Num = len(Signal_SNR)

    Signal_SNR = [int(i.split('_')[1].strip('dB')) for i in Signal_SNR]
    Signal_SNR = list(set(Signal_SNR))
    Signal_SNR.sort()
    Signal_SNR = [str(i)+'dB' for i in Signal_SNR]
    Signal_Num = Signal_Num/len(Signal_SNR)
    # 建立pandas分类矩阵
    result_divid = pd.DataFrame(columns = Signal_SNR, index = Signal_kind)

    print('Signal kind:', Signal_kind)
    print('Signal SNR:', Signal_SNR)
    print('Signal Num:', Signal_Num)
    
    label_total = []
    pre_total = []
    for i in range(len(Signal_kind)):
            for j in range(len(Signal_SNR)):
                print('Signal kind:',Signal_kind[i],'; Signal SNR:',Signal_SNR[j])

                label_divided = []
                pre_divided = []
                for num in range(int(Signal_Num)):
                    file_path = test_data_path + Signal_kind[i] +'/'+ Signal_kind[i]+ '_'+ Signal_SNR[j]+ '_' + str(num+1).zfill(4) + '.jpg'
                    img = image.load_img(file_path, target_size=(img_width, img_height))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
            
                    pre_divided += [np.argmax(model.predict(img))]
                    label_divided += [i]
                
                label_total += label_divided
                pre_total += pre_divided
                
                pre_divided = np.array(pre_divided)
                label_divided = np.array(label_divided)
                acc_divide = np.mean(pre_divided == label_divided)
                result_divid.loc[Signal_kind[i], Signal_SNR[j]] = acc_divide 
    
    print('\n'*2,'-'*10,'divided result','-'*10)
    print(result_divid)

    label_total = np.array(label_total)
    pre_total = np.array(pre_total)
    acc = np.mean(label_total == pre_total)
    print('total accuracy:',acc)
    
    # 计算混淆矩阵
    re = confusion_matrix(label_total , pre_total)
    print('\n'*2,'-'*10,'confuse matrix','-'*10)
    print(re)
    
    # 保存结果
    result_divid.to_csv(result_file)