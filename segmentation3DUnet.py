import sys
import os
import tensorflow as tf
from itertools import product
import numpy as np
import argparse
import yaml
import SimpleITK as sitk
from pathlib import Path
from functions import createParentPath, penalty_categorical, kidney_dice, cancer_dice
from cut import cut3D, caluculate_area, cut_image, padAndCenterCrop, inverse_image
from tqdm import tqdm
import re

args = None

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("imageDirectory", help="Input image file")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.hdf5).")
    parser.add_argument("savePath", help="The filename of the segmented label image")
    parser.add_argument("--patchSize", help="128-128-8", default="128-128-8")
    parser.add_argument("--batchSize", default=1, type=int)
    parser.add_argument("--expandSize", default=15, type=int)
    parser.add_argument("--paddingSize", default=100, type=int)
    parser.add_argument("-g", "--gpuid", help="ID of GPU to be used for segmentation. [default=0]", default=0, type=int)
    args = parser.parse_args()
    return args


def main(_):
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    config.allow_soft_placement = True
#    sess = tf.Session(config=config)
#    tf.keras.backend.set_session(sess)
#
#    # Load model.
#    with tf.device('/device:GPU:{}'.format(args.gpuid)):
#        print("loading 3D U-net model", args.modelweightfile, end="...", flush=True)
#        model = tf.compat.v1.keras.models.load_model(args.modelweightfile, custom_objects={"penalty_categorical" : penalty_categorical, "kidney_dice" : kidney_dice, "caner_dice" : cancer_dice})
#        print("Done.")
#
#    print("input_shape :", model.input_shape)
#    print("output_shape :", model.output_shape)
#
    # ----- Start making patch. -----
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.patchSize)
    if matchobj is None:
        print('[ERROR] Invalid patch size : {}'.format(args.patchSize))
        return
    patchSize  = [ int(s) for s in matchobj.groups() ]
    print("patchSize per patch: ", patchSize)

    labelFile = Path(args.imageDirectory) / 'segmentation.nii.gz'
    imageFile = Path(args.imageDirectory) / 'imaging.nii.gz'

    ## Read image
    label = sitk.ReadImage(str(labelFile))
    image = sitk.ReadImage(str(imageFile))

    labelArray = sitk.GetArrayFromImage(label)
    imageArray = sitk.GetArrayFromImage(image)
    
    labelMinVal = labelArray.min()
    imageMinVal = imageArray.min()

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
    
    restoreMeta= [[] for _ in range(2)]
    stackedImageArrayList = [[] for _ in range(2)]
    stackedLabelArrayList = [[] for _ in range(2)]
    axisCutIndex = [[] for _ in range(2)]
    cutIndex = [[] for _ in range(2)]
    test = [[] for _ in range(2)]
    for i in range(2):
        roiLabelList= []
        roiImageList= []
        
        #一つの腎臓を反転
        if i == 1:
            largestKidneyROILabel[i] = largestKidneyROILabel[i][::-1,:,:]
            largestKidneyROIImage[i] = largestKidneyROIImage[i][::-1,:,:]

        largestKidneyROILabel[i] = np.pad(largestKidneyROILabel[i], [(args.paddingSize, args.paddingSize), (args.paddingSize, args.paddingSize), (0, 0)],"constant", constant_values= labelMinVal)
        largestKidneyROIImage[i] = np.pad(largestKidneyROIImage[i], [(args.paddingSize, args.paddingSize), (args.paddingSize, args.paddingSize), (0, 0)], "constant", constant_values=imageMinVal)

        
        #axial方向について、３D画像として切り取る
        largestKidneyROILabel[i], largestKidneyROIImage[i], cutIndex[i], snum = cut3D(largestKidneyROILabel[i],largestKidneyROIImage[i],"axial")
        
        axialSize = largestKidneyROILabel[i].shape[2]

        ##最大サイズの腎臓を持つスライスの特定
        mArea = []
        for x in range(axialSize):
            mArea.append(caluculate_area(largestKidneyROILabel[i][:,:, x]))
            maxArea = np.argmax(mArea)
        
        #最大サイズのスライスの幅、高さの計算
        maxAreaLabelArray = largestKidneyROILabel[i][..., maxArea]
        roi, maxCenter, maxwh, maxAngle = cut_image(maxAreaLabelArray, paddingSize=expandSize)

        maxWH = (max(maxwh), max(maxwh))

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

                center = maxCenter
                wh = maxWH
                angle = 0

            ##腎臓領域ありの時
            else:
                roi, center, wh, angle = cut_image(largestKidneyROILabel[i][..., x],wh=maxWH, paddingSize=expandSize)##中心,角度取得
                roi_label = roi
                    
                roi, center, wh, angle = cut_image(largestKidneyROIImage[i][..., x], center=center, wh=wh, angle=angle, paddingSize=expandSize)
                roi_image = roi

            restoreMeta[i].append([
                largestKidneyROILabel[i][..., x], 
                center, angle
                ])


            roiLabelList.append(roi_label)
            roiImageList.append(roi_image)

        length = len(roiLabelList)
        imageZero = np.zeros_like(roi_label) + imageMinVal
        labelZero = np.zeros_like(roi_label) + labelMinVal
        
        for x in range(-patchSize[2] + 1, length):
            stackedImageArray = []
            for o in range(patchSize[2]):
                if 0 <= x + o < length:
                    stackedImageArray.append(roiImageList[x + o])

                else:
                    stackedImageArray.append(imageZero)


            #For test
            if x >= 0:
                roiLabelList[x] = padAndCenterCrop(roiLabelList[x], patchSize)
                test[i].append(roiLabelList[x])

            stackedImageArray = np.dstack(stackedImageArray)
            stackedImageArray = padAndCenterCrop(stackedImageArray, patchSize)
            stackedImageArrayList[i].append(stackedImageArray)
            
        # ----- Finish making patch. -----

    # restoreDict -> Meta data for image restoration.
    # stackedImageArrayList -> The list which has patched images.

    # ----- Start segmenting -----
    predictArray = [[] for _ in range(2)]
    for i in range(2):
        #Remove batch dimension
#        outputSize = model.output_shape[1:]
#
#        predictArraySize = outputSize.copy()
#        predictArraySize[2] = len(stackedImageArrayList[i]) + 2 * (predictArraySize[2] - 1)
#        predictArray[i] = np.zeros(predictArraySize)
#
#        patchAxisSize = outputSize[2]
#        length = len(stackedImageArrayList[i])
#        for x, stackedImageArray in enumerate(stackedImageArrayList[i]):
#            stackedImageArray = [np.newaxis, ...]
#            print("Shape of input image : {}".format(stackedImageArray))
#            preArray = model.predict(stackedImageArray, batch_size=args.batchSize)
#            preArray.reshape(outputSize)
#
#            predictArray[i][.., x + patchAxisSize, ...] += preArray
#
#        slices = slice(patchAxisSize - 1, patchAxisSize + length + 1)
#        
#        predictArray[i] = predictArray[..., slices, ...]
#
#        predictArray[i] /= patchAxisSize
#        predictArray[i] = np.argmax(predictArray[i], axis=-1).astype(np.int)

        #For test
        predictArray[i] = np.dstack(test[i])
    # ----- Finish segmenting. -----

    # predictArray -> segmented arary by U-Net.

    # ----- Start restoring image. -----
    outputArray = np.zeros_like(labelArray)
    for i in range(2):
        length = predictArray[i].shape[2]
        restoreImageArrayList = []
        for l in tqdm(range(length), desc="Segmenting...", ncols=60):
            restoreImageArray = inverse_image(predictArray[i][..., l], *restoreMeta[i][l], paddingSize=args.expandSize, clipSize=args.paddingSize)
            restoreImageArrayList.append(restoreImageArray)
        restoreImageArray = np.dstack(restoreImageArrayList)

        slices = slice(cutIndex[i][0], cutIndex[i][-1] + 1)
        if i == 1:
            restoreImageArray = restoreImageArray[::-1, ...]
            outputArray[: secondKidneyIndex, :, slices] = restoreImageArray
        else:
            outputArray[firstKidneyIndex : , :, slices] = restoreImageArray
    # ----- Finish restoring image. -----

    output = sitk.GetImageFromArray(outputArray)
    output.SetDirection(label.GetDirection())
    output.SetOrigin(label.GetOrigin())
    output.SetSpacing(label.GetSpacing())

    print("Saving image to {}...".format(args.savePath))
    sitk.WriteImage(output, args.savePath, True)

    tf.keras.backend.clear_session()

if __name__ == '__main__':
    args = ParseArgs()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]])
