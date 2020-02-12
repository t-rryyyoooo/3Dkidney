import numpy as np
import SimpleITK as sitk

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

def GenerateBatchData(dataList, paddingSize, batch_size = 32):
    ps = paddingSize[::-1] # (x, y, z) -> (z, y, x) for np.array
    #j = 0

    while True:
        indices = list(range(len(dataList)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            imageList = []
            outputList = []

            for idx in indices[i:i+batch_size]:
                image = ImportImage(dataList[idx][0])
                onehotLabel = ImportImage(dataList[idx][1])

                onehotLabel = onehotLabel[ps[0]:-ps[0], ps[1]:-ps[1], ps[2]:-ps[2]]
                imageList.append(image)
                outputList.append(onehotlabel)

            yield (np.array(imageList), np.array(outputList))


