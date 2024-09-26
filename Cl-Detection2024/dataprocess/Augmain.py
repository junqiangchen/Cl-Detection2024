from dataprocess.Augmentation.ImageAugmentation import DataAug3D, DataAug3DClassify
from dataprocess.Augmentation.ImageAugmentation import DataAug, DataAugClassify


def dataaug3dexample():
    aug = DataAug3D(rotation=10, width_shift=0.01, height_shift=0.01, depth_shift=0, zoom_range=0,
                    vertical_flip=True, horizontal_flip=True)
    aug.DataAugmentation('data\Instance/traindata.csv', 5,
                         aug_path='D:\challenge\data\Instance2022/trainstage/augtrain/')


def dataaug2dexample():
    aug = DataAug(rotation=20, width_shift=0.01, height_shift=0.01, vertical_flip=True, horizontal_flip=True)
    aug.DataAugmentation('data/train.csv', 5,
                         aug_path=r'E:\MedicalData\CL-Detection2023\augtrain/')


def classifydataAugB():
    aug = DataAugClassify(rotation=10, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation('data/DRAC/taskBtrain.csv', label=0, number=10,
                         path=r"D:\challenge\data\DRAC2022\process\B\augdata\\")
    aug.DataAugmentation('data/DRAC/taskBtrain.csv', label=1, number=5,
                         path=r"D:\challenge\data\DRAC2022\process\B\augdata\\")


def classifydataAugC():
    aug = DataAugClassify(rotation=10, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation(r'data/DRAC/taskCtrain.csv', label=2, number=4,
                         path=r"D:\challenge\data\DRAC2022\process\C\augdata\\")


def classifydataAugType():
    aug = DataAug3DClassify(rotation=10, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation(r'data\luna22prequel/trainType.csv', label=0, number=10,
                         path=r"D:\challenge\data\2022Lung Nodule Analysis\NoduleType\augtrain\\")
    aug.DataAugmentation(r'data\luna22prequel/trainType.csv', label=1, number=4,
                         path=r"D:\challenge\data\2022Lung Nodule Analysis\NoduleType\augtrain\\")


def classifydataAugMalignancy():
    aug = DataAug3DClassify(rotation=10, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation(r'data\luna22prequel/trainMalignancy.csv', label=0, number=2,
                         path=r"D:\challenge\data\2022Lung Nodule Analysis\NoduleMalignancy\augtrain\\")


def classifydataAugTypeBinary():
    aug = DataAug3DClassify(rotation=10, width_shift=0.01, height_shift=0.01, rescale=1.1)
    aug.DataAugmentation(r'data\luna22prequel/trainTypebinary.csv', label=0, number=10,
                         path=r"D:\challenge\data\2022Lung Nodule Analysis\NoduleTypebinary\augtrain\\")


if __name__ == '__main__':
    dataaug2dexample()
