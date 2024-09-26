import pandas as pd
import torch
import os
from model import *
import numpy as np

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def trainVnet2dlandmarkerbrain():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv(r'dataprocess\\data\\alldata.csv', header=None)
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    csv_data2 = pd.read_csv(r'dataprocess\\data\\alldata.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    vnet2d = VNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=2, loss_name='focal_loss', use_cuda=use_cuda,
                                          accum_gradient_iter=1, num_cpu=2)
    vnet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/1024/vnetfocal_loss', epochs=300,
                        showwind=[8, 7])
    vnet2d.clear_GPU_cache()


def trainUnet2dlandmarkerbrain():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv(r'dataprocess\\data\\train.csv', header=None)
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    csv_data2 = pd.read_csv(r'dataprocess\\data\\valid.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet2d = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=1, loss_name='focal_loss', use_cuda=use_cuda,
                                          accum_gradient_iter=1, num_cpu=2)
    unet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/1024/unetfocal_loss', epochs=300,
                        showwind=[8, 7])
    unet2d.clear_GPU_cache()


def trainUnet2dlandmarkerbrainl2():
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv(r'dataprocess\\data\\train.csv', header=None)
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    trainimages = imagedata[perm]
    trainlabels = maskdata[perm]

    csv_data2 = pd.read_csv(r'dataprocess\\data\\valid.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values

    unet2d = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=1, loss_name='L2', use_cuda=use_cuda,
                                          accum_gradient_iter=1, num_cpu=2)
    unet2d.trainprocess(trainimages, trainlabels, valimages, vallabels, model_dir='log/1024/unetL2', epochs=300,
                        showwind=[8, 7])
    unet2d.clear_GPU_cache()


if __name__ == '__main__':
    trainVnet2dlandmarkerbrain()
    trainUnet2dlandmarkerbrain()
    trainUnet2dlandmarkerbrainl2()
