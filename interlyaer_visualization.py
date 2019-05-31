import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.preprocessing import image
# from keras.applications.resnet50 import preprocess_input



base_model = VGG16()
base_model.summary()

# 载入图片
filepath = './img/me.jpg'
img = image.load_img(filepath, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# 获取指定中间层模型
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_conv2').output)

interlayer_output = model.predict(x)
print(interlayer_output.shape)
n = 5
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    # print(image.shape)
plt.show()

model = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_conv2').output)

interlayer_output = model.predict(x)
print(interlayer_output.shape)
n = 5
for ind in range(n*n):
    image = interlayer_output[0,:,:,ind]
    plt.subplot(n,n,ind+1)
    plt.imshow(image)
    # print(image.shape)
plt.show()

# 获取中间层权重
# interlayer_weights = base_model.get_layer('block2_conv2').get_weights()
# print(interlayer_weights[0].shape)