import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import cv2
import matplotlib.pyplot as plt
from sys import getsizeof
import os

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, Activation, MaxPooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.applications.resnet50 import preprocess_input

import helper_modules

base_model = applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200,200,3), pooling=None)

print(base_model.summary())

for layer in base_model.layers[:-75]:
     layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512,activation='relu')(x)
x = Dense(8, activation='softmax')(x)
model = Model(base_model.input, x)
print("---------------------------------------------------------------------------")
print(model.summary())
from keras import metrics
def top_2_accuracy(y_true, y_pred):
    return metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)
adam = optimizers.Adam(lr=0.000024,beta_1=0.9, beta_2=0.999, epsilon=k.epsilon(), decay=0.0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor = 0.5,patience = 2, min_lr = 0.00001, verbose = 1)
train_directory = "Images/train_images_cropped"
t_size = (200, 200)
b_size = 16
test_directory = "Images/validation_images_cropped_downsampled"
train_gen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip = True
)

test_gen = ImageDataGenerator(
preprocessing_function=preprocess_input,
horizontal_flip = True)

train_generator = train_gen.flow_from_directory(
train_directory,
target_size = t_size,
batch_size = b_size,
class_mode = "categorical")

validation_generator = test_gen.flow_from_directory(
test_directory,
target_size = t_size,
batch_size = b_size,
class_mode = "categorical")


def weighted_categorical_loss(weights):
    weights = k.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= k.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = k.clip(y_pred, k.epsilon(), 1 - k.epsilon())
        # calc
        loss = y_true * k.log(y_pred) * weights
        loss = -k.sum(loss, -1)
        return loss

    return loss

weights = np.array([1.0,1.0,2.0,2.0,3.0,3.0,2.0,3.0])

model.compile(optimizer=adam, loss=weighted_categorical_loss(weights), metrics=['accuracy',top_2_accuracy])

checkpoint = ModelCheckpoint(filepath='saved_models/model.weights.best._new.{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=False)

model.load_weights(filepath="saved_models/model.weights.best.down_sampled_0102.06-0.54.hdf5")

history10 = model.fit_generator(train_generator, epochs=7, validation_data=validation_generator, verbose=2, callbacks=[checkpoint,reduce_lr], steps_per_epoch=200, validation_steps=12)



