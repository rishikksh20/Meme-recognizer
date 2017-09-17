import tensorflow as tf
import model
from PIL import Image
import numpy as np
from keras import backend as K

# Constant seed for replicating training results
np.random.seed(2017)

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("img", "", "Image for classifier")
tf.flags.DEFINE_string("weight", "model_weights.h5", "Weights path")
tf.flags.DEFINE_float("threshold", 0.49, 'Thresholding predicted output')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

img_width= 28
img_height = 28


img = Image.open(FLAGS.img).convert('RGB')
input_img=np.asarray(img.resize((28,28)))
input_img = input_img.astype('float32')
input_img /= 255
if K.image_data_format() == 'channels_first':
    input_img=input_img.reshape((1,3,28, 28))
else:
    input_img=input_img.reshape((1, 28, 28, 3))
dict_={0:'meme',1:'not meme'}
classifier = model.get_model(img_width, img_width)
classifier.load_weights(FLAGS.weight)
y=classifier.predict(input_img)
if y>FLAGS.threshold:
    print(dict_.get(1))
else:
    print(dict_.get(0))
