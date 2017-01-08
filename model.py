'''

Self driving car Naodegree - Udacity

Project 3 : Behavioral Cloning


Reference source :
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

ihangulo@gmail.com

1st DEC 19 2016
2nd JAN 8 2017

'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np
import util
import csv
import cv2

batch_size = 32
nb_epoch = 20


# read csv log file and return two arrays
def read_csv_file(dirname, filename='driving_log.csv', training = True) :
    center_img_fnames = []
    steering_angles = []

    with open(dirname+"/"+filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if(float(row['speed'])>0.0001  or training) :  # remove too low speed data
                this_angle = float(row['steering'])
                center_img_fnames.append(dirname+"/"+row['center'].strip())  # center image name
                steering_angles.append(this_angle) # angle
    return center_img_fnames, steering_angles


# read image file
center_images =[]
y_train = []

center_fnames = []
steering_angles = []

tmp_center = []
tmp_angle = []


tmp_center, tmp_angle = read_csv_file("data_recovery","driving_log.csv", training=False)
center_fnames += tmp_center
steering_angles += tmp_angle


#  split train set --> train set & validation set (85%:15%)
from sklearn.model_selection import train_test_split

X_train_filename, X_test_filename, y_train_angle, y_test_angle = \
    train_test_split(center_fnames, steering_angles,
                                        test_size=0.15,
                                        random_state=10)

#
# read csv log file and return two arrays
#
def generate_arrays_from_array(file_names, angles, batch_size, shuffle=True):
    t_angles = []
    t_filenames =[]
    center_images=[]

    y_train = []

    if(shuffle) : #shuffle
        gen_indices = np.arange(len(angles))
        np.random.shuffle(gen_indices)

        for idx_no in gen_indices:
            t_angles.append(angles[idx_no])
            t_filenames.append(file_names[idx_no])

    else :
        t_angles = angles
        t_filenames = file_names

    while 1:

        for center, angle in zip(t_filenames, t_angles) :

         center_img = cv2.imread(center, cv2.IMREAD_GRAYSCALE)
         center_img = cv2.resize(center_img, (80, 40))  # resize image 80x40 (for fast process)
         center_img = util.crop_driving_image(center_img) # crop 80x20
         center_img = cv2.equalizeHist(center_img) # image processing

         center_images.append(center_img)  # image(training set)
         y_train.append(angle)  # angle (label)

         if(len(center_images) == batch_size) : # batch size

             t_img = util.reformat_driving_image(center_images)

             yield ( (t_img, y_train)  )
             center_images=[] # now reset
             y_train=[]



# Model
from keras.layers.normalization import BatchNormalization

model = Sequential()

model.add(Convolution2D(24, 3, 3, border_mode='same',subsample=(2, 2),
input_shape=(20, 80, 1)))

model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Convolution2D(48, 3, 3,border_mode='same' ,subsample=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(96, 2, 2 ,border_mode='same')) # 64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Convolution2D(192, 2, 2,border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5)) # 0.49 ok

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))


model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(1,activation="linear"))

model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

# https://keras.io/getting-started/faq/#how-can-i-interrupt-training-when-the-validation-loss-isnt-decreasing-anymore
#How can I interrupt training when the validation loss isn't decreasing anymore?
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit_generator(generate_arrays_from_array(
    file_names=X_train_filename ,
    angles = y_train_angle,
    batch_size=batch_size),
         samples_per_epoch= (int)(len(y_train_angle)/batch_size)*batch_size, nb_epoch=nb_epoch,
         callbacks=[early_stopping],
         validation_data = generate_arrays_from_array(
                                    file_names= X_test_filename,
                                    angles = y_test_angle,
                                    batch_size=batch_size, shuffle=False ),
                                    nb_val_samples = 0.15)


import json

#save model as json file
model_json = model.to_json()

with open("model.json", 'w') as f :
    json.dump(model_json,f)
    #save weights
    model.save_weights('model.h5')
    print ("saved")


# added after review1
# show learning history graphs
import matplotlib.pyplot as plt
# http: // machinelearningmastery.com / display - deep - learning - model - training - history - in -keras /
def draw_history_graph(history) :
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
draw_history_graph(history)

#end of program

