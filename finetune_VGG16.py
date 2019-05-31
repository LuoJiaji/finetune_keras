import os
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Flatten

batch_size = 32
epochs = 10
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
for layers in base_model.layers[:num_layers-5]:
     layers.trainable = False
     print(layers.name)
    # print(type(layers))

print('trainable weights:')
for x in base_model.trainable_weights:
    print(x.name)

# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

                     
#img = load_img('./img/me.jpg',target_size=(224,224))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#
#pre = base_model.predict(x)

# 图像Generator，用来构建输入数据
train_datagen = ImageDataGenerator(width_shift_range=0,
                                  height_shift_range=0,
                                  zoom_range=0,
                                  horizontal_flip=False)

train_generator = train_datagen.flow_from_directory('./data_set', 
                                                   target_size = (224, 224), 
                                                   batch_size = batch_size)

image_numbers = train_generator.samples
class_numbers = train_generator.class_indices
print('image numbers:',image_numbers)
print('class',class_numbers)
#
#
x = GlobalAveragePooling2D(name='average_pooling')(base_model.get_layer('block5_conv3').output)
x = Dense(128,name = 'fc_1')(x)
prediction = Dense(5,activation='softmax',name='fc_2')(x)
print(x.shape)
              
model = Model(inputs=base_model.input, outputs=prediction)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

#
##model.fit_generator(train_generator,
##                    steps_per_epoch = image_numbers, 
##                    batch_size, epochs = epochs, 
##                    validation_data = validation_generator, 
##                    validation_steps = batch_size)
# 
#
#model.fit_generator(train_generator,
#                   steps_per_epoch = 20, 
#                   epochs = epochs)


path = './data_set/daisy/'
filename = os.listdir('./data_set/daisy/')

img_list = []
for i in filename[:10]:
    print(i)
    filepath = path+i
    print(filepath)
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    img_list += [x]
x = np.array(img_list)
x = preprocess_input(x)   
pre = model.predict(x)




