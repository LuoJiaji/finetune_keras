
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16

# model = VGG16(include_top=False)
# model = VGG16()
# model = VGG19()
# model = InceptionV3()
model = Xception()
# model = ResNet50()
model.summary()