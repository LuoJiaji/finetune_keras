# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:17:26 2019

@author: Bllue
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input

base_model = ResNet50()
base_model.summary()

# 载入图片
filepath = './dataset8/train/LFM/LFM_6dB_0001.jpg'
img = image.load_img(filepath, target_size=(224, 224))
x = image.img_to_array(img)

plt.figure()
plt.imshow(x[:,:,0])
plt.axis('off')

x = np.expand_dims(x, axis=0)

#x = preprocess_input(x)
x = x/np.max(x)
plt.ion()

# 获取指定中间层模型
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv1').output)
interlayer_output = model.predict(x)
print('conv1 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('conv1')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
conv1 = interlayer_output[0,:,:,:]



model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn_conv1').output)
interlayer_output = model.predict(x)
print('bn_conv1 shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('bn_conv1')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
bn_conv1 = interlayer_output[0,:,:,:]


model = Model(inputs=base_model.input, outputs=base_model.get_layer('res5c_branch2b').output)
interlayer_output = model.predict(x)
print('res5c_branch2b shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('res5c_branch2b')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
res5c_branch2b = interlayer_output[0,:,:,:]



model = Model(inputs=base_model.input, outputs=base_model.get_layer('bn5c_branch2b').output)
interlayer_output = model.predict(x)
print('bn5c_branch2b shape:',interlayer_output.shape)
n = 5
plt.figure()
plt.suptitle('bn5c_branch2b')
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    plt.axis('off')
    # print(image.shape)
plt.show()
bn5c_branch2b = interlayer_output[0,:,:,:]

