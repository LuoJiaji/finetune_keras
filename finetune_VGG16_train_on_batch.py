import os
import random
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.layers.convolutional import Conv2D

batch_size = 128
img_width = 224
img_height = 224

train_dataset_path = './dataset8/train/'
test_data_path = './dataset8/test/'
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
#            datafile.append(os.path.join(datapath,path,file))
            datafile.append(datapath+path+'/'+file)
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
base_model.summary()

# interlayer_weights = base_model.get_layer('block2_conv2').get_weights()
# print(interlayer_weights[0].shape)

# interlayer_weights = base_model.get_layer('block5_conv2').get_weights()
# print(interlayer_weights[0].shape)

# VGG16一共23层
# layers =  base_model.layers
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
x = base_model.get_layer('block5_conv3').output
x = Conv2D(128, (3, 3), strides=(2,2), activation='relu', padding='same', name='block1_CNN')(x)
x = GlobalAveragePooling2D(name='average_pooling')(x)
# x = base_model.get_layer('fc2').output

x = Dense(128,name = 'fc_1')(x)
x = Dense(128,name = 'fc_11')(x)
prediction = Dense(n_calss,activation='softmax',name='fc_2')(x)
print(x.shape)
              
model = Model(inputs=base_model.input, outputs=prediction)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()
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
STEPS = 20000
log_file = './logs/finetune_VGG16_logs.csv'
acc_file = './logs/finetune_VGG16_acc.csv'
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

    if (it+1) % 100 == 0 or it + 1 == STEPS:
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

model.save('./models/keras/finetune_VGG16.h5')    
f_log.close()
f_acc.close()