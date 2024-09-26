from __future__ import print_function, division
import os
import cv2
import numpy as np
from dataprocess.utils import file_name_path
import matplotlib.pyplot as plt

image_dir = "Image"
mask_dir = "Mask"


def preparesampling2dtraindata(datapath, trainImage, trainMask):
    all_files = file_name_path(datapath + '/' + mask_dir, True, False)
    for subsetindex in range(len(all_files)):
        mask_dir1 = all_files[subsetindex]
        image_name = mask_dir1 + '.png'
        image_file = datapath + "/" + image_dir + "/" + image_name
        imagearray = cv2.imread(image_file, 0)
        masklist = []
        for i in range(1, 54, 1):
            mask_gt_file = datapath + '/' + mask_dir + "/" + mask_dir1 + "/" + str(i) + ".png"
            maskarray = cv2.imread(mask_gt_file, 0)
            masklist.append(maskarray)
        masksarray = np.array(masklist)
        masksarray = np.transpose(masksarray, (1, 2, 0))
        # step 3 get subimages and submasks
        if not os.path.exists(trainImage):
            os.makedirs(trainImage)
        if not os.path.exists(trainMask):
            os.makedirs(trainMask)
        filepath1 = trainImage + "\\" + str(mask_dir1) + ".npy"
        filepath = trainMask + "\\" + str(mask_dir1) + ".npy"
        np.save(filepath1, imagearray.astype('uint8'))
        np.save(filepath, masksarray.astype('uint8'))
        print(imagearray.shape, masksarray.shape)
        # # land mark heatmaps showing
        # max_heat = np.sum(masksarray, axis=2)
        # max_heat = 0.4 * imagearray + 0.6 * max_heat
        # plt.figure("heatmap")
        # plt.imshow(max_heat)
        # plt.axis('on')
        # plt.title('heatmap')
        # plt.show()


def preparetraindata():
    """
    :return:
    """
    src_train_path = r"F:\MedicalData\2024CL-Detection\process"
    source_process_path = r"E:\MedicalData\2024CL-Detection\train"
    outputimagepath = source_process_path + "/" + image_dir
    outputlabelpath = source_process_path + "/" + mask_dir
    if not os.path.exists(outputimagepath):
        os.makedirs(outputimagepath)
    if not os.path.exists(outputlabelpath):
        os.makedirs(outputlabelpath)
    preparesampling2dtraindata(src_train_path, outputimagepath, outputlabelpath)


if __name__ == "__main__":
    preparetraindata()
