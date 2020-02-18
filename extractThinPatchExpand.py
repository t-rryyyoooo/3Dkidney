import SimpleITK as sitk
import numpy as np
import argparse
import copy
import os
import sys
from functions import createParentPath, write_file, Resampling, ResampleSize
from cut import *
from pathlib import Path
import re

args = None
'''

This is not complete.

'''

def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("originalFilePath", help="$HOME/Desktop/data/kits19/case_00000")
    parser.add_argument("savePath", help="$HOME/Desktop/data/slice/hist_0.0")
    parser.add_argument("--outputSize", default="128-128-8")
    parser.add_argument("--expandSize", default=15, type=int)
    parser.add_argument("--paddingSize",default=100, type=int)

    args = parser.parse_args()
    return args

def main(args):

    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.outputSize)
    if matchobj is None:
        print('[ERROR] Invalid patch size : {}'.format(args.outputSize))
        return
    outputSize = [ int(s) for s in matchobj.groups() ]
    print("outputSize : ", outputSize)

    labelFile = Path(args.originalFilePath) / 'segmentation.nii.gz'
    imageFile = Path(args.originalFilePath) / 'imaging.nii.gz'

    ## Read image
    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    #----------- For resampling, get direction, spacing, origin in 2D---------------
    extractSliceFilter = sitk.ExtractImageFilter()
    size = list(image.GetSize())
    size[0] = 0
    index = (0, 0, 0)
    extractSliceFilter.SetSize(size)
    extractSliceFilter.SetIndex(index)
    sliceImage = extractSliceFilter.Execute(image)

    #-----------------------------------------------------------------------------

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)
    
    labelMinVal = labelArray.min()
    imageMinVal = imageArray.min()

    print("Whole size: ",labelArray.shape)

    # ------------Extract the maximum region with one kidney.------
    saggitalSize = labelArray.shape[0]
    kidneyEncounter = False
    kidneyStartIndex = []
    kidneyEndIndex = []
    for x in range(saggitalSize):
        is_kidney = (labelArray[x,...] != 0).any()
        if is_kidney and not kidneyEncounter:
            kidneyStartIndex.append(x)
            kidneyEncounter = True

        elif not is_kidney and kidneyEncounter:
            kidneyEndIndex.append(x)
            kidneyEncounter = False

    if len(kidneyStartIndex) != 2:
        print("The patient has horse shoe kidney")
        sys.exit()
    

    print(kidneyStartIndex, kidneyEndIndex)

    largestKidneyROILabel = []#[[1つ目の腎臓の行列],[2つ目の腎臓の行列],..]
    largestKidneyROIImage = []

    firstKidneyIndex = kidneyEndIndex[0]
    secondKidneyIndex = kidneyStartIndex[1]

    expandSize = args.expandSize

    largestKidneyROILabel.append(labelArray[firstKidneyIndex : , ...])
    largestKidneyROILabel.append(labelArray[: secondKidneyIndex, ...])
    largestKidneyROIImage.append(imageArray[firstKidneyIndex : , ...])
    largestKidneyROIImage.append(imageArray[: secondKidneyIndex, ...])

    # ----------------------------------------------------------------

    axialSize = labelArray.shape[2]
    
    for i in range(2):
        roiLabelList= []
        roiImageList= []
        stackedImageArrayList = []
        stackedLabelArrayList = []
        
        #一つの腎臓を反転
        if i == 1:
            largestKidneyROILabel[i] = largestKidneyROILabel[i][::-1,:,:]
            largestKidneyROIImage[i] = largestKidneyROIImage[i][::-1,:,:]

        largestKidneyROILabel[i] = np.pad(largestKidneyROILabel[i], [(args.paddingSize, args.paddingSize), (args.paddingSize, args.paddingSize), (0, 0)], "minimum")
        largestKidneyROIImage[i] = np.pad(largestKidneyROIImage[i], [(args.paddingSize, args.paddingSize), (args.paddingSize, args.paddingSize), (0, 0)], "minimum")

        
        #axial方向について、３D画像として切り取る
        largestKidneyROILabel[i], largestKidneyROIImage[i], cutIndex, snum = cut3D(largestKidneyROILabel[i],largestKidneyROIImage[i],"axial")
        
        print("cutted size_"+str(i)+": ",largestKidneyROILabel[i].shape)

        axialSize = largestKidneyROILabel[i].shape[2]

        ##最大サイズの腎臓を持つスライスの特定
        mArea = []
        for x in range(axialSize):
            mArea.append(caluculate_area(largestKidneyROILabel[i][:,:, x]))
            maxArea = np.argmax(mArea)
        
        #最大サイズのスライスの幅、高さの計算
        maxAreaLabelArray = largestKidneyROILabel[i][..., maxArea]
        roi, maxCenter, maxwh, maxAngle = cut_image(maxAreaLabelArray, paddingSize=expandSize)

        maxWH = maxwh
        reverseMaxWH = tuple(list(maxWH)[::-1])

        # Extract roi per slice
        for x in range(axialSize):
            a = caluculate_area(largestKidneyROILabel[i][:,:,x])
            
            ##腎臓のない領域の画像保存
            if a==0:
                x0 = maxCenter[1] - int((maxWH[1]+15)/2)
                x1 = maxCenter[1] + int((maxWH[1]+15)/2)
                y0 = maxCenter[0] - int((maxWH[0]+15)/2)
                y1 = maxCenter[0] + int((maxWH[0]+15)/2)



                roi_label = largestKidneyROILabel[i][x0 :x1, y0 :y1, x]
                roi_image = largestKidneyROIImage[i][x0 :x1, y0 :y1, x]

            ##腎臓領域ありの時
            else:
                roi, center, wh, angle = cut_image(largestKidneyROILabel[i][..., x])
                if (maxWH[0] > maxWH[1]) == (wh[0] > wh[1]):
                    roi, center, wh, angle = cut_image(largestKidneyROILabel[i][..., x], wh=maxWH)
                    roi_label = roi

                else:
                    roi, center, wh, angle = cut_image(largestKidneyROILabel[i][..., x], wh=reverseMaxWH)
                    roi_label = roi

                roi, center, wh, angle = cut_image(largestKidneyROILabel[i][..., x], center=center, wh=wh, angle=angle)
                roi_image = roi

                roi_label = sitk.GetImageFromArray(roi_label)
                roi_image = sitk.GetImageFromArray(roi_image)
                d = sliceImage.GetDirection()
                s = sliceImage.GetSpacing()
                o = sliceImage.GetOrigin()
                roi_label.SetDirection(d)
                roi_label.SetSpacing(s)
                roi_label.SetOrigin(o)
                roi_image.SetDirection(d)
                roi_image.SetSpacing(s)
                roi_image.SetOrigin(o)

                roi_label = ResampleSize(roi_label, outputSize[:2], is_label=True)
                roi_image = ResampleSize(roi_image, outputSize[:2])

                roi_label = sitk.GetArrayFromImage(roi_label)
                roi_image = sitk.GetArrayFromImage(roi_image)


            roiLabelList.append(roi_label)
            roiImageList.append(roi_image)


        length = len(roiLabelList)
        imageZero = np.zeros_like(roi_label) - imageMinVal
        labelZero = np.zeros_like(roi_label) - labelMinVal
        
        for x in range(-outputSize[2] + 1, length):
            stackedImageArray = []
            stackedLabelArray = []
            for o in range(outputSize[2]):
                if 0 <= x + o < length:
                    stackedImageArray.append(roiImageList[x + o])
                    stackedLabelArray.append(roiLabelList[x + o])
                else:
                    stackedImageArray.append(imageZero)
                    stackedLabelArray.append(labelZero)

            stackedImageArray = np.dstack(stackedImageArray)
            stackedLabelArray = np.dstack(stackedLabelArray)
            stackedImageArrayList.append(stackedImageArray)
            stackedLabelArrayList.append(stackedLabelArray)


        patientID = args.originalFilePath.split('/')[-1]
        OPI = Path(args.savePath) / 'image' / patientID / "dummy.mha"
        OPL = Path(args.savePath) / 'image' / patientID / "dummy.mha"
        OPT = Path(args.savePath) / 'path' / (patientID + '.txt')
        #Make parent path
        if not OPI.parent.exists():
            createParentPath(str(OPI))
        
        if not OPL.parent.exists():
            createParentPath(str(OPL))

        if not OPT.parent.exists():
            createParentPath(str(OPT))

        for x in range(length):
            OPI = Path(args.savePath) / 'image' / patientID / "image{}_{:02d}.mha".format(i,x)
            OPL = Path(args.savePath) / 'image' / patientID / "label{}_{:02d}.mha".format(i,x)
            OPT = Path(args.savePath) / 'path' / (patientID + '.txt')
           
            stackedImage = sitk.GetImageFromArray(stackedImageArrayList[x])
            stackedLabel = sitk.GetImageFromArray(stackedLabelArrayList[x])

            stackedImage.SetSpacing(image.GetSpacing())
            stackedImage.SetOrigin(image.GetOrigin())
            stackedImage.SetDirection(image.GetDirection())

            stackedLabel.SetSpacing(image.GetSpacing())
            stackedLabel.SetOrigin(image.GetOrigin())
            stackedLabel.SetDirection(image.GetDirection())


            stackedNewSize = outputSize[::-1]

            stackedImage = ResampleSize(stackedImage, stackedNewSize)
            stackedLabel = ResampleSize(stackedLabel, stackedNewSize, is_label=True)

            sitk.WriteImage(stackedImage, str(OPI), True)
            sitk.WriteImage(stackedLabel, str(OPL), True)
            print("{}/{} done.".format(x + 1, length))

            write_file(str(OPT), str(OPI) + "\t" + str(OPL))
          


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
