import random
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

def ReadSliceDataList(fileName):
    dataList = []
    with open(fileName) as f:
        for line in f:
            imageFile, labelFile = line.strip().split()
            dataList.append((imageFile, labelFile))

    return dataList

def ImportImage(filename):
    image = sitk.ReadImage(filename)
    imageArray = sitk.GetArrayFromImage(image)
    if image.GetNumberOfComponentsPerPixel() == 1:
        imageArray = imageArray[..., np.newaxis]
    return imageArray

def getInputShape(fileName):
    dataList = ReadSliceDataList(fileName)
    imageArray = ImportImage(dataList[0][0])

    return imageArray.shape 

def GetMinimumValue(image):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(image)
    return minmax.GetMinimum()

def Affine(t, r, scale, shear, c):
    a = sitk.AffineTransform(3)

    a.SetCenter(c)
    a.Rotate(1, 0, r)
    a.Shear(1, 0, shear[0])
    a.Shear(0, 1, shear[1])
    a.Scale((scale, scale, 1))
    a.Translate(t)

    return a

def Transforming(image, bspline, affine, interpolator, minval):
    # B-spline transformation
    if bspline is not None:
        transformed_b = sitk.Resample(image, bspline, interpolator, minval)

    # Affine transformation
        transformed_a = sitk.Resample(transformed_b, affine, interpolator, minval)

    else:
        transformed_a = sitk.Resample(image, affine, interpolator, minval)

    return transformed_a

def makeAffineParameters(image, translationRange, rotateRange, shearRange, scaleRange):
    translation = np.random.uniform(-translationRange, translationRange, 3)
    rotation = np.radians(np.random.uniform(-rotateRange, rotateRange))
    shear = np.random.uniform(-shearRange, shearRange, 2)
    scale = np.random.uniform(1-scaleRange, 1+scaleRange)
    center = (np.array(image.GetSize()) * np.array(image.GetSpacing()) / 2)[::-1]
    
    
    return [translation, rotation, scale, shear, center]

def ImportTransformedImage(imageFile, labelFile):
    sigma = 0
    translationrange = 0 # [mm]
    rotrange = 15 # [deg]
    shearrange = 0
    scalerange = 0.05
    bspline = None
    
    image = sitk.ReadImage(imageFile)
    label = sitk.ReadImage(labelFile)
    
    parameters = makeAffineParameters(image, translationrange, rotrange, shearrange, scalerange)
    affine = Affine(*parameters)

    imageMinVal = GetMinimumValue(image)
    labelMinVal = GetMinimumValue(label)

    transformedImage = Transforming(image, bspline, affine, sitk.sitkLinear, imageMinVal)
    transformedLabel = Transforming(label, bspline, affine, sitk.sitkNearestNeighbor, labelMinVal)

    imageArray = sitk.GetArrayFromImage(transformedImage)
    labelArray = sitk.GetArrayFromImage(transformedLabel)
    
    if image.GetNumberOfComponentsPerPixel() == 1:
        imageArray = imageArray[..., np.newaxis]

    if label.GetNumberOfComponentsPerPixel() == 1:
        labelArray = labelArray[..., np.newaxis]
    
    return (imageArray, labelArray)






def GenerateBatchData(dataList, apply_augmentation=False, num_class=3, batch_size = 32):
    while True:
        indices = list(range(len(dataList)))
        random.shuffle(indices)

        if apply_augmentation:
            for i in range(0, len(indices), batch_size):
                imageLabelList = np.array([ImportTransformedImage(dataList[idx][0], dataList[idx][1]) for idx in indices[i : i + batch_size]])

                imageList, labelList = zip(*imageLabelList)
                
                onehotLabelList = tf.keras.utils.to_categorical(labelList, num_classes=num_class)

                yield (np.array(imageList), np.array(onehotLabelList))

        else:
            for i in range(0, len(indices), batch_size):
                imageList = np.array([ImportImage(dataList[idx][0]) for idx in indices[i : i + batch_size]])

                onehotLabelList = np.array([tf.keras.utils.to_categorical(ImportImage(dataList[idx][1], num_classes=num_class)) for idx in indices[i : i + batch_size]])

                yield (imageList, onehotLabelList)


