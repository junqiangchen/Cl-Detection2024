import numpy as np
import torch
import os
from model import *
import cv2
import pandas as pd

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def validVnet2dlandmarkerbrain():
    vnet2d = VNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=2, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=4,
                                          inference=True, model_path=r"log/1024/vnetfocal_loss/BinaryVNet2dModel.pth")
    outputdir = r"F:\MedicalData\2024CL-Detection\validpd\vnet"
    csv_data2 = pd.read_csv(r'dataprocess\\data\\valid.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    distancelist = []
    for i in range(len(valimages)):
        src = np.load(valimages[i])
        maskgt = np.load(vallabels[i])
        coords_array_pd, coords_array_gt, max_heat_pd, max_heat_gt = vnet2d.inferencevalid(src, maskgt)
        distancelist.append(np.mean(np.linalg.norm(coords_array_pd - coords_array_gt, axis=1)))
        max_heat_pd = np.clip(max_heat_pd * 255., 0, 255).astype('uint8')
        max_heat_gt = np.clip(max_heat_gt * 255., 0, 255).astype('uint8')
        cv2.imwrite(outputdir + '/' + str(i) + "image.png", src)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkgt.png", max_heat_gt)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkpd.png", max_heat_pd)
    distancediff = np.mean(np.array(distancelist))
    print("mean distance value:", distancediff)


def validUnet2dlandmarkerbrain():
    unet2d = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=1, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=4,
                                          inference=True, model_path=r"log/1024/unetfocal_loss/BinaryUNet2dModel.pth")
    outputdir = r"F:\MedicalData\2024CL-Detection\validpd\unet"
    csv_data2 = pd.read_csv(r'dataprocess\\data\\valid.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    distancelist = []
    for i in range(len(valimages)):
        src = np.load(valimages[i])
        maskgt = np.load(vallabels[i])
        coords_array_pd, coords_array_gt, max_heat_pd, max_heat_gt = unet2d.inferencevalid(src, maskgt)
        distancelist.append(np.mean(np.linalg.norm(coords_array_pd - coords_array_gt, axis=1)))
        max_heat_pd = np.clip(max_heat_pd * 255., 0, 255).astype('uint8')
        max_heat_gt = np.clip(max_heat_gt * 255., 0, 255).astype('uint8')
        cv2.imwrite(outputdir + '/' + str(i) + "image.png", src)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkgt.png", max_heat_gt)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkpd.png", max_heat_pd)
    distancediff = np.mean(np.array(distancelist))
    print("mean distance value:", distancediff)


def validaVNetUnet2dlandmarkerbrain():
    unet2d = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=1, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=4,
                                          inference=True, model_path=r"log/1024/unetfocal_loss/BinaryUNet2dModel.pth")
    vnet2d = VNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=2, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=4,
                                          inference=True, model_path=r"log/1024/vnetfocal_loss/BinaryVNet2dModel.pth")
    unet2d2 = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                           batch_size=1, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=4,
                                           inference=True, model_path=r"log/1024/unetL2/BinaryUNet2dModel.pth")
    outputdir = r"F:\MedicalData\2024CL-Detection\validpd\vnet+unet"
    csv_data2 = pd.read_csv(r'dataprocess\\data\\valid.csv', header=None)
    valimages = csv_data2.iloc[:, 0].values
    vallabels = csv_data2.iloc[:, 1].values
    distancelist = []
    for i in range(len(valimages)):
        src = np.load(valimages[i])
        maskgt = np.load(vallabels[i])
        imageresize = (src - src.mean()) / src.std()
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        label = np.transpose(maskgt, (2, 0, 1))
        label = label / 255.

        heatmaps_array1 = unet2d.predict(imageresize)
        heatmaps_array2 = vnet2d.predict(imageresize)
        heatmaps_array3 = unet2d2.predict(imageresize)

        heatmaps_array = heatmaps_array1 + heatmaps_array2 + heatmaps_array3
        coords_array_pd, _ = vnet2d.get_landmarks(heatmaps_array)
        coords_array_gt, _ = vnet2d.get_landmarks(label)
        coords_array_pd = np.around(coords_array_pd).astype('int')
        coords_array_gt = np.around(coords_array_gt).astype('int')
        max_heat_pd = np.zeros_like(src)
        max_heat_gt = np.zeros_like(src)
        for num in range(coords_array_pd.shape[0]):
            cv2.circle(max_heat_pd, (coords_array_pd[num][1], coords_array_pd[num][0]), 3, 255, -1)
        for num in range(coords_array_gt.shape[0]):
            cv2.circle(max_heat_gt, (coords_array_gt[num][1], coords_array_gt[num][0]), 3, 255, -1)
        distancelist.append(np.mean(np.linalg.norm(coords_array_pd - coords_array_gt, axis=1)))
        max_heat_pd = np.clip(max_heat_pd * 255., 0, 255).astype('uint8')
        max_heat_gt = np.clip(max_heat_gt * 255., 0, 255).astype('uint8')
        cv2.imwrite(outputdir + '/' + str(i) + "image.png", src)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkgt.png", max_heat_gt)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkpd.png", max_heat_pd)
    distancediff = np.mean(np.array(distancelist))
    print("mean distance value:", distancediff)


if __name__ == '__main__':
    # validVnet2dlandmarkerbrain()
    # validUnet2dlandmarkerbrain()
    validaVNetUnet2dlandmarkerbrain()
