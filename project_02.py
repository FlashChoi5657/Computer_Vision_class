import os, mh_loader, shutil, mh_overlay
import SimpleITK as sitk
import numpy as np

from Net.Unet2d import unet_2d
import tensorflow as tf
import Net.Unet3d as unet_3d

####################
# Preprocessing
####################

path='/home/check5657/Dataset/01_POSTECH_Ankle_MRI'

data, label = list(), list()

for case in os.listdir(f'{path}/data'):

    obj1 = mh_loader.series_load(f'{path}/data/{case}/T1 2mm')
    arr1 = sitk.GetArrayFromImage(obj1)
    arr1 = arr1[..., np.newaxis]
    obj2 = mh_loader.series_load(f'{path}/data/{case}/T2 2mm')
    arr2 = sitk.GetArrayFromImage(obj2)
    arr2 = arr2[..., np.newaxis]
    arr = np.concatenate((arr1,arr2),-1)

    # lab1 = sitk.ReadImage(f'{path}/backup/{case}_T1_axial_2mm.nii')
    lab1 = sitk.ReadImage(f'{path}/label/{case}/{case}.nii')
    lab_arr = sitk.GetArrayFromImage(lab1)
    a, b = np.unique(lab_arr, return_counts=True)
    for i in range(len(a)):
        lab_arr = np.where(lab_arr == a[i], int(i),lab_arr)

    lab_arr = lab_arr[..., np.newaxis]

    # if lab.GetSize()[2] != obj.GetSize()[2]:
    #     print(case, obj.GetSize(), lab.GetSize())

    for i in range(arr.shape[0]):
        new_img = arr[i,...]
        data.append(new_img)
    for j in range(lab_arr.shape[0]):
        new_label = lab_arr[j,...]
        label.append(new_label)

##################
# data preprocessing
#################

from keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rotation_range=25, width_shift_range=0.3, height_shift_range=0.3, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest', rescale=1.0/255.0)
val_gen = ImageDataGenerator(rescale=1/255.0)

rlr_call = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor = 0.1, patience = 5, verbose = 1)
es_call = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience = 7, verbose = 1)

flow_train_gen = train_gen.flow(x=np.array(data)[:-16], y=np.array(label)[:-16], batch_size=4, shuffle=True)
flow_val_gen = val_gen.flow(x=np.array(data)[-16:], y=np.array(label)[-16:], batch_size=4, shuffle=True)

train_x = np.array(data)[:-16]
train_y = np.array(label)[:-16]
val_x = np.array(data)[-16:]
val_y = np.array(data)[-16:]

##############################
# Network Architecture
################################

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, hinge, squared_hinge


def class_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + hinge(y_true, y_pred) + squared_hinge(y_true, y_pred)

def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.greater(y_pred, .5), tf.float32)
    return (2. * tf.reduce_sum(y_true * y_pred) + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def class_dice_loss(y_true, y_pred):
    return class_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


##################################
# Training
###############################

nb_filter = [32 * 2 ** i for i in range(5)]
unet2d = unet_2d((512,512,2),nb_filter, 8, unpool=False, weights=None)
# unet2d.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
unet2d.compile(optimizer=Adam(4e-3), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[dice_coef])
# tf.keras.utils.plot_model(unet2d, to_file='model.png', show_shapes=True)
unet2d.summary(150)


model_history = unet2d.fit_generator(flow_train_gen, epochs = 5, validation_data = flow_val_gen, callbacks=[rlr_call, es_call])
# unet2d.fit(train_x, train_y, batch_size = 4, epochs = 5,validation_data=(val_x,val_y))

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

###################
# test
###################


test1 = mh_loader.series_load(f'{path}/test/FC4D5480D29B/T1 2mm')
test2 = mh_loader.series_load(f'{path}/test/FC4D5480D29B/T2 2mm')
test1 = sitk.GetArrayFromImage(test1)
test2 = sitk.GetArrayFromImage(test2)
test1 = test1[..., np.newaxis]
test2 = test2[..., np.newaxis]
test = np.concatenate((test1,test2), -1)

prediction = unet2d.predict(test)

prediction[0]






