from __future__ import print_function, division
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pandas as pd


def remove_zero_padding(image_array: np.ndarray) -> np.ndarray:
    """
    function to remove zero padding in an image | 去除图像中的0填充函数
    :param image_array: one cephalometric image array, shape is (2400, 2880) | 一张头影图像的矩阵，形状为(2400, 2880)
    :return: image matrix after removing zero padding | 去除零填充部分的图像矩阵
    """
    row = np.sum(image_array, axis=1)
    column = np.sum(image_array, axis=0)

    non_zero_row_indices = np.argwhere(row != 0)
    non_zero_column_indices = np.argwhere(column != 0)

    last_row = int(non_zero_row_indices[-1])
    last_column = int(non_zero_column_indices[-1])

    image_array = image_array[:last_row + 1, :last_column + 1]
    return image_array


def save_image_landmarks(srcimg, seg_image, index, trainImage, trainMask):
    if os.path.exists(trainImage) is False:
        os.mkdir(trainImage)
    if os.path.exists(trainMask) is False:
        os.mkdir(trainMask)
    filepath = trainImage + "\\" + str(index) + ".png"
    cv2.imwrite(filepath, srcimg)
    filepathdir = trainMask + "\\" + str(index)
    if os.path.exists(filepathdir) is False:
        os.mkdir(filepathdir)
    for lid in range(np.shape(seg_image)[0]):
        filepath2 = filepathdir + "\\" + str(lid + 1) + ".png"
        seg_mask = np.clip(seg_image[lid] * 255., 0, 255).astype('uint8')
        cv2.imwrite(filepath2, seg_mask)


def onelandmarktoheatmap(srcimage, coords, sigma, sigma_scale_factor=1.0, size_sigma_factor=10, normalize_center=True):
    """
    Generates a numpy array of the landmark image for the specified point and parameters.
    :param srcimage:input src image
    :param coords:one landmark coords on src image([x], [x, y] or [x, y, z]) of the point.
    :param sigma:Sigma of Gaussian
    :param sigma_scale_factor:Every value of the gaussian is multiplied by this value.
    :param size_sigma_factor:the region size for which values are being calculated
    :param normalize_center:if true, the value on the center is set to scale_factor
                             otherwise, the default gaussian normalization factor is used
    :return:heatmapimage
    """
    # landmark holds the image
    srcimage = np.squeeze(srcimage)
    image_size = np.shape(srcimage)
    assert len(image_size) == len(coords), "image dim is not equal landmark coords dim"
    dim = len(coords)
    heatmap = np.zeros(image_size, dtype=float)
    # flip point is form [x, y, z]
    flipped_coords = np.array(coords)
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    # check the region start and region end size is in the image range
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)
    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap
    region_size = (region_end - region_start).astype(int)
    sigma = sigma * sigma_scale_factor
    scale = 1.0
    if not normalize_center:
        scale /= math.pow(math.sqrt(2 * math.pi) * sigma, dim)
    if dim == 1:
        dx = np.meshgrid(range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        squared_distances = x_diff * x_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0]] = cropped_heatmap[:]
    if dim == 2:
        dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        squared_distances = x_diff * x_diff + y_diff * y_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1]] = cropped_heatmap[:, :]
    if dim == 3:
        dy, dx, dz = np.meshgrid(range(region_size[1]), range(region_size[0]), range(region_size[2]))
        x_diff = dx + region_start[0] - flipped_coords[0]
        y_diff = dy + region_start[1] - flipped_coords[1]
        z_diff = dz + region_start[2] - flipped_coords[2]
        squared_distances = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
        cropped_heatmap = scale * np.exp(-squared_distances / (2 * math.pow(sigma, 2)))
        heatmap[region_start[0]:region_end[0], region_start[1]:region_end[1],
        region_start[2]:region_end[2]] = cropped_heatmap[:, :, :]
    return heatmap


def LandmarkGeneratorHeatmap(srcimage, lanmarks, sigma=3.0):
    """
    Generates a numpy array landmark images for the specified points and parameters.
    :param srcimage:src image itk
    :param lanmarks:image landmarks array
    :param sigma:Sigma of Gaussian
    :return:heatmap
    """
    heatmap_list = []
    for landmark in lanmarks:
        heatmap_list.append(onelandmarktoheatmap(srcimage, landmark, sigma))
    heatmaps = np.stack(heatmap_list, axis=0)
    ## land mark heatmaps showing
    # max_heat = np.sum(heatmaps, axis=0)
    # max_heat = 0.4 * srcimage + 0.6 * max_heat * 255
    # plt.figure("heatmap")
    # plt.imshow(max_heat)
    # plt.axis('on')
    # plt.title('heatmap')
    # plt.show()
    return heatmaps


def preparedata():
    path = r"F:\MedicalData\2024CL-Detection\download\CL-Detection2024 Accessible Data\Training Set\images"
    label_path = r"F:\MedicalData\2024CL-Detection\download\CL-Detection2024 Accessible Data\Training Set\labels.csv"
    trainImage = r"F:\MedicalData\2024CL-Detection\process\Image"
    trainMask = r"F:\MedicalData\2024CL-Detection\process\Mask"
    sigma = 10
    newSize = (1024, 1024)
    # step 1 load the landmark and image from file
    csv_data = pd.read_csv(label_path)
    imagefiles = csv_data.iloc[:, 0].values
    spacingsvalues = csv_data.iloc[:, 1].values
    landmarksvalues = csv_data.iloc[:, 2:].values
    for indx in range(len(imagefiles)):
        # step1 load one image
        sliceimage = cv2.imread(path + '/' + imagefiles[indx], 0)
        # step2 resize image to fixed image and get scale x and scale y
        scale_y, scale_x = newSize[0] / sliceimage.shape[0], newSize[1] / sliceimage.shape[1]
        sliceimage = cv2.resize(sliceimage, dsize=newSize)
        # step3 load landmarks and rescale coor
        landmarksvalue = landmarksvalues[indx]
        landmarks = []
        for labelid in range(0, len(landmarksvalue) - 1, 2):
            x_coor, y_coor = landmarksvalue[labelid], landmarksvalue[labelid + 1]
            x_coor = x_coor * scale_x
            y_coor = y_coor * scale_y
            landmarks.append([y_coor, x_coor])
        # step 5 generate landmarks heatmaps
        heatmaps = LandmarkGeneratorHeatmap(sliceimage, landmarks, sigma=sigma)
        # step 6 save image and heatmaps image to file
        save_image_landmarks(sliceimage, heatmaps, indx + 1, trainImage, trainMask)


if __name__ == '__main__':
    preparedata()
