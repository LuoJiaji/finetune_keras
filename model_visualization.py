# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:49:51 2018

@author: Bllue
"""
import cv2
import numpy as np
from keras.preprocessing import image
#from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

from keras.layers import Dense, GlobalAveragePooling2D, Flatten,Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
import keras.backend as K

#from keras.applications.resnet50 import preprocess_input
image_path = './dataset8/test/Barker/Barker_0dB_0001.jpg'
image_path = './dataset8/train/Barker/Barker_0dB_0001.jpg'

img1 = image.load_img(image_path, target_size=(224, 224))
img1 = image.img_to_array(img1)
#x = np.expand_dims(x, axis=0)
img1 = preprocess_input(img1)
img1 = np.expand_dims(img1, axis=0)

img2 = cv2.imread(image_path)
img2 = cv2.resize(img2,(600,600))
img2 = np.expand_dims(img2, axis=0)


base_model = VGG16()
plot_model(base_model, to_file='./model_visualization/VGG16.pdf',show_shapes=True)
#model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.get_layer('bn5c_branch2c').output))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
preds_VGG16 = model.predict(img1)

base_model = VGG19()
plot_model(base_model, to_file='./model_visualization/VGG19.pdf',show_shapes=True)
#model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.get_layer('bn5c_branch2c').output))
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
preds_VGG19 = model.predict(img1)

#base_model = ResNet50()
##model = Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.get_layer('bn5c_branch2c').output))
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('activation_49').output)
#preds_Res50 = model.predict(img1)

#plot_model(base_model, to_file='./model_visualization/ResNet50.pdf',show_shapes=True)
#plot_model(base_model, to_file='./model_visualization/ResNet50.png',show_shapes=True)

#Inp = Input((300, 300, 3))
#base_model = ResNet50(input_shape=(300,300,3))
#x = base_model(Inp)
##y = base_model.get_layer('bn5c_branch2c').output

#model = Model(inputs=Inp, outputs=x)

#base_model = InceptionV3()
##base_model.summary()
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed10').output)
###model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)
#preds_Inception = model.predict(img2)
#plot_model(base_model, to_file='./model_visualization/InceptionV3.pdf',show_shapes=True)

#base_model = Xception()
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block14_sepconv2_bn').output)
##model = Model(inputs=base_model.input, outputs=base_model.get_layer('flatten_1').output)
#preds_Xception = model.predict(img2)
#plot_model(base_model, to_file='./model_visualization/Xception.pdf',show_shapes=True)