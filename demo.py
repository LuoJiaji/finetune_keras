import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image




base_model = VGG16()
base_model.summary()

interlayer_weights = base_model.get_layer('block2_conv2').get_weights()
print(interlayer_weights[0].shape)

interlayer_weights = base_model.get_layer('block5_conv2').get_weights()
print(interlayer_weights[0].shape)

# VGG16一共23层
# layers =  base_model.layers
num_layers = len(base_model.layers)
print('number of layers:',num_layers)

print('layer name:') 
for layers in base_model.layers:
    # layers.trainable = False
    print(layers.name)
    # print(type(layers))

print('trainable weights:')
for x in base_model.trainable_weights:
    print(x.name)

# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)