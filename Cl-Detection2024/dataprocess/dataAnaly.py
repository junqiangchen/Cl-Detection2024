from __future__ import print_function, division
import SimpleITK as sitk
import numpy as np
from dataprocess.utils import file_name_path

imagetype = "mask.nii.gz"


def getImageSizeandSpacing(aorticvalve_path):
    """
    get image and spacing
    :return:
    """
    file_path_list = file_name_path(aorticvalve_path, False, True)
    size = []
    spacing = []
    for subsetindex in range(len(file_path_list)):
        if imagetype in file_path_list[subsetindex]:
            mask_name = file_path_list[subsetindex]
            mask_gt_file = aorticvalve_path + "/" + mask_name
            src = sitk.ReadImage(mask_gt_file, sitk.sitkUInt8)
            imageSize = src.GetSize()
            imageSpacing = src.GetSpacing()
            size.append(np.array(imageSize))
            spacing.append(np.array(imageSpacing))
            print("image size,image spacing:", (imageSize, imageSpacing))
    print("mean size,mean spacing:", (np.mean(np.array(size), axis=0), np.mean(np.array(spacing), axis=0)))
    print("median size,median spacing:", (np.median(np.array(size), axis=0), np.median(np.array(spacing), axis=0)))
    print("min size,min spacing:", (np.min(np.array(size), axis=0), np.min(np.array(spacing), axis=0)))
    print("max size,max spacing:", (np.max(np.array(size), axis=0), np.max(np.array(spacing), axis=0)))


if __name__ == "__main__":
    aorticvalve_path = r"F:\MedicalData\synthRAD2023\roiprocess\task2\pelvis"
    getImageSizeandSpacing(aorticvalve_path)
