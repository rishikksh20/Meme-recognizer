from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import model
from keras import backend as K
import numpy as np
from pathlib import Path

# Constant seed for replicating training results
np.random.seed(2017)

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_boolean("retrain", False, "Retraining")
tf.flags.DEFINE_integer("batch_size", "16", "Batch Size")
tf.flags.DEFINE_integer("epochs", "50", "Number of epochs")
tf.flags.DEFINE_string("weight", "model_weights.h5", "Weights path")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

train_data_dir = './data/train'
validation_data_dir = './data/validation'
nb_train_samples = 267
nb_validation_samples = 84
epochs = FLAGS.epochs
batch_size = FLAGS.batch_size
img_width = 28
img_height = 28

retrain = FLAGS.retrain

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

# automagically retrieve images and their classes for train and validation sets
train_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

classifier = model.get_model(img_width, img_height)


if retrain:
    if Path(FLAGS.weight).is_file():
        classifier.load_weights(FLAGS.weight)

classifier.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

print("Done !!!")
classifier.save_weights(FLAGS.weight)
K.clear_session()
