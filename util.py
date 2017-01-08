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
import cv2



def crop_driving_image (img) :
    return img[14:34, :]


def reformat_driving_image(dataset):
    np_images_set = np.array(dataset)
    new_images_set_norm = np_images_set/255
    return np.reshape(new_images_set_norm, (-1, 20, 80, 1)).astype(np.float32)



def show_gray_image(img) :
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_gray_image_plt(img) :
    newImg = np.dstack((img, img, img))
    plt.imshow(newImg, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    plt.show()

def show_one_image(img):
    plt.figure(figsize=(1, 1))

    if (img.shape[2] == 1):  # if gray image
        newImg = np.dstack((img, img, img))
        plt.imshow(newImg, cmap=plt.get_cmap('gray'), vmin=-1, vmax=1)
    else:
        plt.imshow(img, cmap=plt.get_cmap)
    plt.show()




# Not used ---------------------------------

    #
    # def change_to_normalize(images_set):
    #     np_images_set = np.array(images_set)
    #     new_images_set_norm = np_images_set / (255)  # normalize
    #     return new_images_set_norm


# # change to grey scale & normalize (0-1)
# # https://codedump.io/share/AbxxuPPXSXZQ/1/how-can-i-convert-an-rgb-image-into-grayscale-in-python
# def change_RGB_to_gray_normalize(images_set):
#     new_images_set = []
#     for image in images_set :
#         new_images_set.append (np.dot(image[...,:3],[0.2989, 0.5870, 0.1140]) / 255.0)
#     return new_images_set
# #
# def change_BGR_to_gray_normalize(images_set, bgr=True):
#    new_images_set = []
#    for image in images_set :
#        if(bgr) :
#            new_images_set.append (np.dot(image[...,:3],[0.1140, 0.5870, 0.2989 ]) / (255.0))
#        else : # RGB
#
#            new_images_set.append(np.dot(image[..., :3], [0.6, 0.2, 0.2]) / 255.0)
#
#    return np.array(new_images_set, dtype="float32")


# return image flipped (vertical)
# def make_image_flip(img, angle) :
#     flipped_image =  np.fliplr(img)
#     flipped_label = -1.0 * angle
#     return flipped_image, flipped_label


# def make_image_arrays_flip (imgs, angles) :
#     if(len(imgs)>1) :
#         for i, img, angle in zip( range(len(imgs)), imgs, angles) :
#             if(np.random.choice(10) == 0 ) : #10%
#                 imgs[i], angles[i]  = make_image_flip(img,angle) # flip
#         return imgs, angles
#     else : # only one
#         return make_image_flip(imgs, angles)


# from datetime import datetime
# def get_diff_time_from_fname(fname1, fname2):
#     # http://stackoverflow.com/questions/466345/converting-string-into-datetime
#
#     fdate1 = fname1[25:-8]  # "IMG/center_2016_12_17_02_42_25_044.jpg" --> 2016_12_17_02_42_25 extract
#     fdate2 = fname2[25:-8]
#
#     # print(fdate1)
#     date1= datetime.strptime(fdate1, "%Y_%m_%d_%H_%M_%S")  # 특정 형식의 문자열을 시간 데이터로 바꾼다
#     date2= datetime.strptime(fdate2, "%Y_%m_%d_%H_%M_%S")  # 특정 형식의 문자열을 시간 데이터로 바꾼다
#     start = datetime(1970, 1, 1)
#
#     sec1= (date1 - start).total_seconds()
#     sec2= (date2 - start).total_seconds()
#
#     return sec2-sec1  # return diff(sec)
#
# def crop_images(dataset) :
#     newdata = []
#     for data in dataset :
#         data =  data[30:68, :, :].copy()
#         newdata.append(data)
#     return  newdata   # half size (80x160 --> 50x160]

#
# def crop_images_small(dataset) :
#
#     newdata = []
#     for data in dataset :
#         ndata =  data[14:34, :, :].copy()
#         newdata.append(ndata)
#     return  newdata   # half size (80x160 --> 50x160]
#
# def crop_images_small_grayscale(dataset, one=False) :
#
#     #print("len=", len(dataset))
#     if(one) :
#         return dataset[14:34,:]  # 20 x 80
#     else :
#         newdata = []
#         for data in dataset :
#             ndata =  data[14:34,:].copy()
#             #ndata = data[:, 14:34].copy()
#             newdata.append(ndata)
#     return  newdata   # half size (80x160 --> 50x160]
#
# # crop 30 lines
# def crop_image(img) :
#
#     return img[30:68,:,  :].copy() # (80x160) -> 38 x 160

# def reformat_one_channel(dataset, label, bgr=False):
#     dataset = crop_images(dataset)
#
#
#     #print ('length=' ,len(dataset))
#
#     if(bgr) :
#
#         dataset = change_BGR_to_gray_normalize(dataset, True)
#         #dataset, label = make_image_arrays_flip(dataset, label)  # make random flip
#
#     else :
#
#         dataset = change_RGB_to_gray_normalize(dataset, False)
#
#
#     dataset  = np.reshape(dataset, (-1, 38, 160, 1)).astype(np.float32) #  38( 50),160
#
#     return dataset, label
#
#
# def reformat_one_channel_small(dataset, bgr=False, one = False):
#     #dataset = crop_images_small_grayscale(dataset, one)
#
#     if(bgr) :
#
#         dataset = change_BGR_to_gray_normalize(dataset, True)
#         #dataset, label = make_image_arrays_flip(dataset, label)  # make random flip
#
#     else : # RGB to grayscale
#
#         #dataset = change_BGR_to_gray_normalize(dataset, False)
#         dataset =   change_to_normalize(dataset) #
#
#
#     dataset  = np.reshape(dataset, (-1, 20, 80, 1)).astype(np.float32) #  38( 50),160
#     #dataset = np.reshape(dataset, (-1, 20,80, 3)).astype(np.float32)  # 38( 50),160
#
#
#     return dataset