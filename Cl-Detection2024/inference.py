import numpy as np
import torch
import os
from model import *
import cv2
import pandas as pd
from dataprocess.utils import file_name_path

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
use_cuda = torch.cuda.is_available()


def inferenceUnetVnet2dlandmarkerbrain():
    vnet2d = VNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=2, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=2,
                                          inference=True,
                                          model_path=r"log/1024/vnetfocal_loss/BinaryVNet2dModelbest.pth")
    unet2d = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                          batch_size=1, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=2,
                                          inference=True,
                                          model_path=r"log/1024/unetfocal_loss/BinaryUNet2dModelbest.pth")
    unet2d2 = UNet2dlandmarkdetectionModel(image_height=1024, image_width=1024, image_channel=1, numclass=53,
                                           batch_size=1, loss_name='focal_loss', accum_gradient_iter=1, num_cpu=2,
                                           inference=True, model_path=r"log/1024/unetL2/BinaryUNet2dModelbest.pth")
    all_images_predict_landmarks_list = []
    all_images_list = []
    newSize = (1024, 1024)
    datadir = r"F:\MedicalData\2024CL-Detection\download\CL-Detection2024 Accessible Data\Validation Set\images"
    outputdir = r"F:\MedicalData\2024CL-Detection\download\CL-Detection2024 Accessible Data\Validation Set\Mask"
    all_image_files = file_name_path(datadir, False, True)
    for i in range(len(all_image_files)):
        sliceimage = cv2.imread(datadir + '/' + all_image_files[i], 0)
        all_images_list.append(all_image_files[i])
        # step2 resize image to fixed size
        imageresize = cv2.resize(sliceimage, newSize, interpolation=cv2.INTER_LINEAR)
        imageresize = (imageresize - imageresize.mean()) / imageresize.std()
        offsety, offsetx = newSize[0] / sliceimage.shape[0], newSize[1] / sliceimage.shape[1]
        # transpose (H,W,C) order to (C,H,W) order
        H, W = np.shape(imageresize)[0], np.shape(imageresize)[1]
        imageresize = np.reshape(imageresize, (H, W, 1))
        imageresize = np.transpose(imageresize, (2, 0, 1))
        # step 3 get heatmaps array
        heatmaps_array1 = vnet2d.predict(imageresize)
        heatmaps_array2 = unet2d.predict(imageresize)
        heatmaps_array3 = unet2d2.predict(imageresize)
        # step 4 add heatmaps and get result
        heatmaps_array = heatmaps_array1 + heatmaps_array2 + heatmaps_array3
        coords_array_pd, _ = vnet2d.get_landmarks(heatmaps_array)
        # step5 resize landmarks_coords to src image size
        coords_array_pd = coords_array_pd * np.array((1 / offsety, 1 / offsetx))
        # step 6 save resize landmarks to json file and draw on image
        landmarks_list = []
        for num in range(coords_array_pd.shape[0]):
            landmarks_list.append([coords_array_pd[num][1], coords_array_pd[num][0]])
        all_images_predict_landmarks_list.append(landmarks_list)

        round_coords_array_pd = np.round(coords_array_pd).astype('int')
        max_heat_pd = cv2.cvtColor(sliceimage, cv2.COLOR_GRAY2BGR)
        for num in range(round_coords_array_pd.shape[0]):
            cv2.circle(max_heat_pd, (round_coords_array_pd[num][1], round_coords_array_pd[num][0]), 5, (0, 0, 255), -1)
        cv2.imwrite(outputdir + '/' + str(i) + "landmarkpd1024.png", max_heat_pd)
        # cv2.imwrite(outputdir + '/' + str(i) + "image.png", sliceimage)

    data = [
        ["image file", "p1x", "p1y", "p2x", "p2y", "p3x", "p3y", "p4x", "p4y", "p5x", "p5y", "p6x", "p6y", "p7x", "p7y",
         "p8x", "p8y", "p9x", "p9y", "p10x", "p10y", "p11x", "p11y", "p12x", "p12y", "p13x", "p13y", "p14x", "p14y",
         "p15x", "p15y", "p16x", "p16y", "p17x", "p17y", "p18x", "p18y", "p19x", "p19y", "p20x", "p20y", "p21x", "p21y",
         "p22x", "p22y", "p23x", "p23y", "p24x", "p24y", "p25x", "p25y", "p26x", "p26y", "p27x", "p27y", "p28x", "p28y",
         "p29x", "p29y", "p30x", "p30y", "p31x", "p31y", "p32x", "p32y", "p33x", "p33y", "p34x", "p34y", "p35x", "p35y",
         "p36x", "p36y", "p37x", "p37y", "p38x", "p38y", "p39x", "p39y", "p40x", "p40y", "p41x", "p41y", "p42x", "p42y",
         "p43x", "p43y", "p44x", "p44y", "p45x", "p45y", "p46x", "p46y", "p47x", "p47y", "p48x", "p48y", "p49x", "p49y",
         "p50x", "p50y", "p51x", "p51y", "p52x", "p52y", "p53x", "p53y"
         ]]
    for i in range(len(all_images_list)):
        oneimagedata = []
        image_name = all_image_files[i]
        oneimagedata.append(image_name)
        image_landmarks = all_images_predict_landmarks_list[i]
        for j in range(len(image_landmarks)):
            oneimagedata.append(round(image_landmarks[j][0]))
            oneimagedata.append(round(image_landmarks[j][1]))
        data.append(oneimagedata)
    # 将列表转换为 DataFrame
    df = pd.DataFrame(data[1:], columns=data[0])
    # 将 DataFrame 写入 CSV 文件
    df.to_csv('predictions20240625.csv', index=False)
    print("Data has been written to output.csv")


if __name__ == '__main__':
    inferenceUnetVnet2dlandmarkerbrain()
