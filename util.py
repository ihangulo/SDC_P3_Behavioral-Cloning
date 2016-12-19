'''

Self driving car Naodegree - Udacity

Project 3 : Behavioral Cloning

util.py

Reference source :
    https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

ihangulo@gmail.com DEC 19 2016
'''


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# return image flipped (vertical)
def make_image_flip(img, angle) :
    flipped_image =  np.fliplr(img)
    flipped_label = -1.0 * angle
    return flipped_image, flipped_label


def make_image_arrays_flip (imgs, angles) :
    if(len(imgs)>1) :
        for i, img, angle in zip( range(len(imgs)), imgs, angles) :
            if(np.random.choice(10) == 0 ) : #10%
                imgs[i], angles[i]  = make_image_flip(img,angle) # flip
        return imgs, angles
    else : # only one
        return make_image_flip(imgs, angles)


def get_diff_time_from_fname(fname1, fname2):
    # http://stackoverflow.com/questions/466345/converting-string-into-datetime

    fdate1 = fname1[25:-8]  # "IMG/center_2016_12_17_02_42_25_044.jpg" --> 2016_12_17_02_42_25 extract
    fdate2 = fname2[25:-8]

    # print(fdate1)
    date1= datetime.strptime(fdate1, "%Y_%m_%d_%H_%M_%S")  # 특정 형식의 문자열을 시간 데이터로 바꾼다
    date2= datetime.strptime(fdate2, "%Y_%m_%d_%H_%M_%S")  # 특정 형식의 문자열을 시간 데이터로 바꾼다
    start = datetime(1970, 1, 1)

    sec1= (date1 - start).total_seconds()
    sec2= (date2 - start).total_seconds()

    return sec2-sec1  # return diff(sec)



def crop_images(dataset) :

    newdata = []
    for data in dataset :
        data =  data[30:68, :, :].copy()
        newdata.append(data)
    return  newdata   # half size (80x160 --> 50x160]

# crop 30 lines
def crop_image(img) :

    return img[30:68,:,  :].copy() # (80x160) -> 38 x 160

def reformat_one_channel(dataset, label, bgr=True):
    dataset = crop_images(dataset)


    print ('length=' ,len(dataset))

    if(bgr) :
        dataset = change_BGR_to_gray_normalize(dataset, True)
        dataset, label = make_image_arrays_flip(dataset, label)  # make random flip

    else :
        dataset = change_BGR_to_gray_normalize(dataset, False)


    dataset  = np.reshape(dataset, (-1, 38, 160, 1)).astype(np.float32) #  38( 50),160

    return dataset, label



def show_one_image(img):
    plt.figure(figsize=(1, 1))
    plt.axis('off')

    if (img.shape[2] == 1):  # if gray image
        newImg = np.dstack((img, img, img))
        plt.imshow(newImg, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    else:
        plt.imshow(img)
    plt.show()


# change to grey scale & normalize (0-1)
# https://codedump.io/share/AbxxuPPXSXZQ/1/how-can-i-convert-an-rgb-image-into-grayscale-in-python
def change_RGB_to_gray_normalize(images_set):
    new_images_set = []
    for image in images_set :
        new_images_set.append (np.dot(image[...,:3],[0.2989, 0.5870, 0.1140]) / 255.0)
    return new_images_set

def change_BGR_to_gray_normalize(images_set, bgr=True):
   new_images_set = []
   for image in images_set :
       if(bgr) :
           new_images_set.append (np.dot(image[...,:3],[0.1140, 0.5870, 0.2989 ]) / (255.0))
       else : # RGB
           new_images_set.append(np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0)

   return np.array(new_images_set, dtype="float32")

#
# def fake_change_BGR_to_gray_normalize(images_set,bgr=True):
#     new_images_set = []
#     for image in images_set:
#         if (bgr): # g b increase
#             new_images_set.append(np.dot(image[..., :3], [0.2, 0.4, 0.4]) / (255.0))
#         else:  # RGB
#             new_images_set.append(np.dot(image[..., :3], [0.4, 0.4, 0.2]) / 255.0)
#
#     return np.array(new_images_set, dtype="float32")




# keyboard wheel angle normalization (mean with next angle)
# def make_smooth_angle(fnames, angles) :
#     # i=0;
#     # while( i <  len(angles)-2) :
#     #     if(angles[i]!= 0 and angles[i+1]==0 and ( get_diff_time_from_fname(fnames[i], fnames[i+1]) < 2) ):
#     #         angles[i+1] = angles[i] / 3.0 # add some variation value
#     #         i += 2
#     #         print("#",end='')
#     #     else :
#     #         i += 1
#     return angles

#
#
# def change_to_gray_normalize (image_Set) :
#     return np.mean(image_Set, axis=3, keepdims=True) / (255.0 /2 ) - 1  # [-1 ... 1 ]
#
# def convert_one_gray_normalize (image) :
#     return np.mean(image, axis=-1, keepdims=True) / (255.0 /2 ) - 1  # [-1 ... 1 ]

#
# def reformat_image_angle(img, angle) :
#
#     img = crop_image(img)
#     convert_one_gray_normalize (img)
#     img, angle = make_image_flip(img, angle)  # make random flip
#     #dataset = np.reshape(dataset, (-1, 50, 160, 1)).astype(np.float32)  # 자르면 50,160
#
#     return img, angle