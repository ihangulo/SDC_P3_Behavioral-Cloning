'''

Self driving car Naodegree - Udacity

Project 3 : Behavioral Cloning


Reference source :
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

ihangulo@gmail.com DEC 19 2016
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import numpy as np
import util
import csv
import cv2

batch_size = 32
nb_epoch = 5



# read csv log file and return two arrays
def read_csv_file(dirname, filename='driving_log.csv') :
    center_img_fnames = []
    steering_angles = []

    with open(dirname+"/"+filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            center_img_fnames.append(dirname+"/"+row['center'].strip())  # center image name
            steering_angles.append(float(row['steering'])) # angle
    return center_img_fnames, steering_angles




# read image file
center_images =[]
y_train = []

center_fnames = []
steering_angles = []

tmp_center = []
tmp_angle = []


# read traning data csv file
tmp_center, tmp_angle = read_csv_file("data_hangulo3","driving_log.csv")
center_fnames += tmp_center
steering_angles += tmp_angle

# read recovery data csv file
tmp_center, tmp_angle = read_csv_file("data_recover","driving_log.csv")
center_fnames += tmp_center
steering_angles += tmp_angle

# read real images & pre process
for center, angle in zip (center_fnames,  steering_angles) :
    center_img = cv2.imread(center) # read BGR image (center camera)
    center_img = cv2.resize(center_img, (160,80))  # resize image 160x80 (for fast process)
    center_images.append(center_img) # image(training set)
    y_train.append(angle) # angle (label)

center_images = np.array(center_images, dtype="float32")
print ("Center img shape", center_images.shape)

y_train = np.array(y_train, dtype="float32")

X_train, y_train = util.reformat_one_channel(center_images, y_train)
print("after", X_train.shape , len(y_train))

#  split train set --> train set & validation set (90%:10%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                                test_size=0.1,
                                                                random_state=0)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# Model

model = Sequential()

model.add(Convolution2D(24, 5, 5, border_mode='same',subsample=(2, 2), #stride
input_shape=X_train.shape[1:]))
model.add(Activation('relu'))


model.add(Convolution2D(36, 5, 5,subsample=(2, 2))) #stride))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))


model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(1))

model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['accuracy'])

history= model.fit(X_train, y_train,
               batch_size=batch_size,
               nb_epoch=nb_epoch,
               validation_data=(X_test, y_test),
               shuffle=True)

import json
#save model as json file
model_json = model.to_json()

with open("model.json", 'w') as f :
    json.dump(model_json,f)
    #save weights
    model.save_weights('model.h5')
    print ("saved")

#end of program