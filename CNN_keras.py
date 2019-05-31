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

batch_size = 32
epochs = 10
dataset_path = './data_set'

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(width_shift_range=0,
                                  height_shift_range=0,
                                  zoom_range=0,
                                  horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(dataset_path, 
                                                   target_size = (224, 224), 
                                                   batch_size = batch_size)

image_numbers = train_generator.samples
class_numbers = train_generator.class_indices
print('image numbers:',image_numbers)
print('class',class_numbers)


input_shape = (224, 224, 3)
input_tensor=Input(shape=input_shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Classification block
# x = Flatten(name='flatten')(x)
x = GlobalAveragePooling2D(name = 'Average_pooling')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(64, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(5, activation='softmax', name='predictions')(x)

model = Model(input = input_tensor, output = x)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.summary()

model.fit_generator(train_generator,
                   steps_per_epoch = 64, 
                   epochs = epochs)