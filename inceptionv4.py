from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import argparse
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils.training_utils import multi_gpu_model
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.utils import shuffle
import os
import random
import IPython.display as display
import scipy
from tensorflow import set_random_seed
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import pathlib
import PIL
import pandas
from keras.layers import concatenate
import subprocess
from time import time
from keras.callbacks import TensorBoard


# Arguments

ap = argparse.ArgumentParser()

ap.add_argument("-g", "--gpus", default=1, type=int,
	help="# of available GPUs")
ap.add_argument("-train", "--train_dir", type=str, default="train/",
    help="train directory")
ap.add_argument("-val", "--val_dir", type=str, default="val/",
	help="val directory")
ap.add_argument("-c", "--checkpoint", type=str, default="no")
ap.add_argument("-classes", "--num_classes", type=int, required=True)
ap.add_argument("-epochs", "--epochs", type=int, default=1000)
ap.add_argument("-steps", "--steps_per_epoch", type=int, default=500)
ap.add_argument("-lr", "--learning_rate", type=float, default='1e-3')
ap.add_argument("-log", "--log_dir", type=str, default="logs/")
args = vars(ap.parse_args())


# Check whether continue training from pretrained model or not

if str(args['checkpoint']) != 'no':
    checkpoint_path = str(args['checkpoint'])
    check = True
else:
    check = False



def conv_block(x, nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1), bias=False):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    x = Convolution2D(nb_filter, nb_row, nb_col, subsample=subsample, border_mode=border_mode, bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def inception_stem(input):

    channel_axis = -1

    # Shape 299 x 299 x 3 
    x = conv_block(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv_block(x, 32, 3, 3, border_mode='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x2 = conv_block(x, 96, 3, 3, subsample=(2, 2), border_mode='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, border_mode='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 96, 3, 3, border_mode='valid')
    x = concatenate([x1, x2], axis=channel_axis)

    x1 = conv_block(x, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    x2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    x = concatenate([x1, x2], axis=channel_axis)

    return x


def inception_A(input):
    channel_axis = -1

    a1 = conv_block(input, 96, 1, 1)

    a2 = conv_block(input, 64, 1, 1)
    a2 = conv_block(a2, 96, 3, 3)

    a3 = conv_block(input, 64, 1, 1)
    a3 = conv_block(a3, 96, 3, 3)
    a3 = conv_block(a3, 96, 3, 3)

    a4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    a4 = conv_block(a4, 96, 1, 1)

    m = concatenate([a1, a2, a3, a4], axis=channel_axis)

    return m


def inception_B(input):
    channel_axis = -1

    b1 = conv_block(input, 384, 1, 1)

    b2 = conv_block(input, 192, 1, 1)
    b2 = conv_block(b2, 224, 1, 7)
    b2 = conv_block(b2, 256, 7, 1)

    b3 = conv_block(input, 192, 1, 1)
    b3 = conv_block(b3, 192, 7, 1)
    b3 = conv_block(b3, 224, 1, 7)
    b3 = conv_block(b3, 224, 7, 1)
    b3 = conv_block(b3, 256, 1, 7)

    b4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    b4 = conv_block(b4, 128, 1, 1)

    m = concatenate([b1, b2, b3, b4], axis=channel_axis)

    return m


def inception_C(input):
    channel_axis = -1

    c1 = conv_block(input, 256, 1, 1)

    c2 = conv_block(input, 384, 1, 1)
    c2_1 = conv_block(c2, 256, 1, 3)
    c2_2 = conv_block(c2, 256, 3, 1)

    c2 = concatenate([c2_1, c2_2], axis=channel_axis)


    c3 = conv_block(input, 384, 1, 1)
    c3 = conv_block(c3, 448, 3, 1)
    c3 = conv_block(c3, 512, 1, 3)
    c3_1 = conv_block(c3, 256, 1, 3)
    c3_2 = conv_block(c3, 256, 3, 1)

    c3 = concatenate([c3_1, c3_2], axis=channel_axis)

    c4 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c4 = conv_block(c4, 256, 1, 1)

    m = concatenate([c1, c2, c3, c4], axis=channel_axis)

    return m


def reduction_A(input):
    channel_axis = -1

    r1 = conv_block(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 192, 1, 1)
    r2 = conv_block(r2, 224, 3, 3)
    r2 = conv_block(r2, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def reduction_B(input):
    channel_axis = -1

    r1 = conv_block(input, 192, 1, 1)
    r1 = conv_block(r1, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    r2 = conv_block(input, 256, 1, 1)
    r2 = conv_block(r2, 256, 1, 7)
    r2 = conv_block(r2, 320, 7, 1)
    r2 = conv_block(r2, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(input)

    m = concatenate([r1, r2, r3], axis=channel_axis)

    return m


def create_inception_v4(nb_classes=int(args["num_classes"]), load_weights=check):

    init = Input((299,299, 3))

    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8, 8))(x)

    # Dropout - Use 0.2, as mentioned in official paper. 
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')

    if check == True:
        weights = checkpoint_path
        model.load_weights(weights)
        print("Model weights loaded.")

    return model

model = create_inception_v4(load_weights=False)

if int(args['gpus']) > 1:
    model = multi_gpu_model(model, gpus=int(args['gpus']))

model.summary()

train_dir = str(args['train_dir'])
val_dir = str(args['val_dir'])
subprocess.run('mkdir inceptionv4_checkpoints', shell=True)
print('-----------------------------')
print('$ # of GPUs - ', str(args['gpus']))
print('$ # of Classes - ', str(args['num_classes']))
print('$ Learning rate - ', str(args['learning_rate']))
print('$ Epochs ', str(args['epochs']))
print('-----------------------------')


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

datagen=ImageDataGenerator(rescale=1/255,
            rotation_range=40,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest',
            samplewise_std_normalization=True)

val_datagen = ImageDataGenerator(rescale=1/255)

train_generator = datagen.flow_from_directory(train_dir,target_size=(299,299),class_mode="categorical")
val_gen = datagen.flow_from_directory(val_dir,target_size=(299,299),class_mode="categorical")

mc = keras.callbacks.ModelCheckpoint("inceptionv4_checkpoints/InceptionV4.h5",save_best_only=True, save_weights_only=True)
tensorboard = TensorBoard(log_dir="{}/{}".format(args["log_dir"], time()))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=float(args['learning_rate']), decay=1e-6, momentum=0.9, nesterov=True), metrics=["accuracy"])
hist = model.fit_generator(train_generator,steps_per_epoch=int(args['steps_per_epoch']),epochs=int(args['epochs']),verbose=True,validation_data=val_gen,validation_steps=10,callbacks=[mc, tensorboard])