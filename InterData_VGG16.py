# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:17:26 2019

@author: Bllue
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.layers.normalization import BatchNormalization
s
# from keras.applications.resnet50 import preprocess_input

base_model = VGG16()
base_model.summary()

# 载入图片
filepath = './dataset8/train/LFM/LFM_6dB_0001.jpg'
img = image.load_img(filepath, target_size=(224, 224))
x = image.img_to_array(img)

plt.figure()
plt.imshow(x[:,:,0])
plt.axis('off')

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
#x = x/np.max(x)
plt.ion()

# 获取指定中间层模型
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv2').output)
interlayer_output = model.predict(x)
print('block1_conv2 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block1_conv2')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block1_conv2 = interlayer_output[0,:,:,:]

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').output)
interlayer_output = model.predict(x)
print('block1_pool shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block1_pool')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block1_pool = interlayer_output[0,:,:,:]


model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)
interlayer_output = model.predict(x)
print('block2_conv2 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block2_conv2')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block2_conv2 = interlayer_output[0,:,:,:]

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').output)
interlayer_output = model.predict(x)
print('block2_pool shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block2_pool')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block2_pool = interlayer_output[0,:,:,:]


model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_conv3').output)
interlayer_output = model.predict(x)
print('block3_conv3 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block3_conv3')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block3_conv3 = interlayer_output[0,:,:,:]

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').output)
interlayer_output = model.predict(x)
print('block3_pool shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block3_pool')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block3_pool = interlayer_output[0,:,:,:]


model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)
interlayer_output = model.predict(x)
print('block4_conv3 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block4_conv3')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block4_conv3 = interlayer_output[0,:,:,:]

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
interlayer_output = model.predict(x)
print('block4_pool shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block4_pool')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block4_pool = interlayer_output[0,:,:,:]



model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)
interlayer_output = model.predict(x)
print('block5_conv3 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block5_conv3')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()

block5_conv3 = interlayer_output[0,:,:,:]

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
interlayer_output = model.predict(x)
print('block5_pool shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('block5_pool')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
block5_pool = interlayer_output[0,:,:,:]