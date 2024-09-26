from ..Augmentation.images_masks_3dtransform import ImageDataGenerator3D
import numpy as np
from dataprocess.Augmentation.images_masks_transform import ImageDataGenerator
import pandas as pd
import os
import cv2

'''
Feature Standardization
standardize pixel values across the entire dataset
ZCA Whitening
A whitening transform of an image is a linear algebra operation that reduces the redundancy in the matrix of pixel images.
Less redundancy in the image is intended to better highlight the structures and features in the image to the learning algorithm.
Typically, image whitening is performed using the Principal Component Analysis (PCA) technique.
More recently, an alternative called ZCA (learn more in Appendix A of this tech report) shows better results and results in
transformed images that keeps all of the original dimensions and unlike PCA, resulting transformed images still look like their originals.
Random Rotations
sample data may have varying and different rotations in the scene.
Random Shifts
images may not be centered in the frame. They may be off-center in a variety of different ways.
RESCALE
对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0- 1之间，通常为1 / 255;
Random Flips
improve performance on large and complex problems is to create random flips of images in your training data.
fill_mode: 填充像素, 出现在旋转或平移之后．
'''


class DataAug(object):
    '''
    transform Image and Mask together
    '''

    def __init__(self, rotation=5, width_shift=0.05,
                 height_shift=0.05, rescale=1.2, horizontal_flip=True, vertical_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=rotation, width_shift_range=width_shift,
                                            height_shift_range=height_shift, rescale=rescale,
                                            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                            fill_mode='nearest')

    def __ImageMaskTranform(self, images, labels, index, number, path):
        # reshape to be [samples][pixels][width][height][channels]
        images = np.load(images)
        labels = np.load(labels)
        if len(images.shape) == 2:
            srcimage = images.reshape([1, images.shape[0], images.shape[1], 1])
            srclabel = labels.reshape([1, labels.shape[0], labels.shape[1], -1])
        else:
            srcimage = images.reshape([1, images.shape[0], images.shape[1], images.shape[2]])
            srclabel = labels.reshape([1, labels.shape[0], labels.shape[1], labels.shape[2]])

        i = 0
        for batch1, batch2 in self.__datagen.flow(srcimage, srclabel):
            i += 1
            batch1 = batch1[0]
            src_path = path + 'Image\\'
            if not os.path.exists(src_path):
                os.makedirs(src_path)
            np.save(src_path + str(index) + '_' + str(i) + '.npy', batch1.astype('uint8'))

            batch2 = batch2[0]
            mask_path = path + 'Mask\\'
            if not os.path.exists(mask_path):
                os.makedirs(mask_path)
            np.save(mask_path + str(index) + '_' + str(i) + '.npy', batch2.astype('uint8'))

            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, number=100, aug_path=None):
        csvdata = pd.read_csv(filepathX)
        dataX = csvdata.iloc[:, 0].values
        dataY = csvdata.iloc[:, 1].values
        for index in range(dataX.shape[0]):
            # For images
            images = dataX[index]
            # For labels
            labels = dataY[index]
            self.__ImageMaskTranform(images, labels, index, number, aug_path)


class DataAugClassify(object):
    '''
    transform Image and Mask together
    '''

    def __init__(self, rotation=5, width_shift=0.05,
                 height_shift=0.05, rescale=1.2, horizontal_flip=True, vertical_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=rotation, width_shift_range=width_shift,
                                            height_shift_range=height_shift, rescale=rescale,
                                            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip,
                                            fill_mode='nearest')

    def __ImageMaskTranform(self, images, labels, index, number, path):
        # reshape to be [samples][pixels][width][height][channels]
        images = cv2.imread(images, 0)
        if len(images.shape) == 2:
            srcimage = images.reshape([1, images.shape[0], images.shape[1], 1])
        else:
            srcimage = images.reshape([1, images.shape[0], images.shape[1], images.shape[2]])

        i = 0
        for batch1, _ in self.__datagen.flow(srcimage, srcimage):
            i += 1
            batch1 = batch1[0]
            src_path = path + str(labels) + '\\'
            if not os.path.exists(src_path):
                os.makedirs(src_path)
            cv2.imwrite(src_path + str(index) + '_' + str(i) + '.bmp', batch1)
            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, label, number=100, path=None):
        csvdata = pd.read_csv(filepathX)
        dataX = csvdata.iloc[:, 1].values
        dataY = csvdata.iloc[:, 0].values
        for index in range(dataX.shape[0]):
            # For images
            images = dataX[index]
            # For labels
            labels = dataY[index]
            if labels == label:
                self.__ImageMaskTranform(images, labels, index, number, path)


class DataAug3D(object):
    '''
        transform Image and Mask together
        '''

    def __init__(self, rotation=5, width_shift=0.01, height_shift=0.01, depth_shift=0.01, zoom_range=0.01,
                 rescale=1.1, horizontal_flip=True, vertical_flip=False, depth_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator3D(rotation_range=rotation, width_shift_range=width_shift,
                                              height_shift_range=height_shift, depth_shift_range=depth_shift,
                                              zoom_range=zoom_range,
                                              rescale=rescale, horizontal_flip=horizontal_flip,
                                              vertical_flip=vertical_flip, depth_flip=depth_flip,
                                              fill_mode='nearest')

    def __ImageMaskTranform(self, images_path, index, number, maxvalue=255):
        # reshape to be [samples][depth][width][height][channels]
        imagesample = np.load(images_path[0])
        srcimages = np.zeros((imagesample.shape[0], imagesample.shape[1], imagesample.shape[2]))
        srcimage = imagesample.reshape([1, srcimages.shape[0], srcimages.shape[1], srcimages.shape[2], 1])

        masksample = np.load(images_path[1])
        srcmasks = np.zeros((masksample.shape[0], masksample.shape[1], masksample.shape[2]))
        srcmask = masksample.reshape([1, srcmasks.shape[0], srcmasks.shape[1], srcmasks.shape[2], 1])

        i = 0
        for batchx, batchy in self.__datagen.flow(srcimage, srcmask):
            i += 1
            batch1 = batchx[0, :, :, :, :]
            batch2 = batchy[0, :, :, :, :]

            npy_path1 = self.aug_path + 'Image/'
            if not os.path.exists(npy_path1):
                os.makedirs(npy_path1)
            npy_path1 = npy_path1 + str(index) + '_' + str(i) + ".npy"
            batch1 = batch1.reshape([srcimages.shape[0], srcimages.shape[1], srcimages.shape[2]])
            np.save(npy_path1, batch1)

            npy_path2 = self.aug_path + 'Mask/'
            if not os.path.exists(npy_path2):
                os.makedirs(npy_path2)
            npy_path2 = npy_path2 + str(index) + '_' + str(i) + ".npy"
            batch2 = batch2.reshape([srcmasks.shape[0], srcmasks.shape[1], srcmasks.shape[2]])
            # batch2 = batch2.astype('uint8')
            np.save(npy_path2, batch2)
            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, number=100, aug_path=None):
        csvXdata = pd.read_csv(filepathX)
        data = csvXdata.iloc[:, :].values
        self.aug_path = aug_path
        for index in range(data.shape[0]):
            # For images
            images_path = data[index]
            self.__ImageMaskTranform(images_path, index, number)


class DataAug3DClassify(object):
    '''
        transform Image and Mask together
        '''

    def __init__(self, rotation=5, width_shift=0.01, height_shift=0.01, depth_shift=0.01, zoom_range=0.01,
                 rescale=1.1, horizontal_flip=True, vertical_flip=False, depth_flip=False):
        # define data preparation
        self.__datagen = ImageDataGenerator3D(rotation_range=rotation, width_shift_range=width_shift,
                                              height_shift_range=height_shift, depth_shift_range=depth_shift,
                                              zoom_range=zoom_range,
                                              rescale=rescale, horizontal_flip=horizontal_flip,
                                              vertical_flip=vertical_flip, depth_flip=depth_flip,
                                              fill_mode='nearest')

    def __ImageMaskTranform(self, images, labels, index, number, path):
        # reshape to be [samples][depth][width][height][channels]
        imagesample = np.load(images)
        srcimages = np.zeros((imagesample.shape[0], imagesample.shape[1], imagesample.shape[2]))
        srcimage = imagesample.reshape([1, srcimages.shape[0], srcimages.shape[1], srcimages.shape[2], 1])

        i = 0
        for batch1, _ in self.__datagen.flow(srcimage, srcimage):
            i += 1
            batch1 = batch1[0]
            src_path = path + str(labels) + '\\'
            if not os.path.exists(src_path):
                os.makedirs(src_path)
            npy_path1 = src_path + '/' + str(index) + '_' + str(i) + ".npy"
            batch1 = batch1.reshape([srcimages.shape[0], srcimages.shape[1], srcimages.shape[2]])
            np.save(npy_path1, batch1)
            if i > number - 1:
                break

    def DataAugmentation(self, filepathX, label, number=100, path=None):
        csvdata = pd.read_csv(filepathX)
        dataX = csvdata.iloc[:, 1].values
        dataY = csvdata.iloc[:, 0].values
        for index in range(dataX.shape[0]):
            # For images
            images = dataX[index]
            # For labels
            labels = dataY[index]
            if labels == label:
                self.__ImageMaskTranform(images, labels, index, number, path)
