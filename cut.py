import numpy as np
import cv2
import SimpleITK as sitk
from functions import Resampling, ResamplingInAxial

def cut3D(labelArray, imageArray, axis):
    if axis=="axial":
        totalNumber = len(labelArray[0,0,:])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[:,:,x]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = int(len(imageIndex)/10)

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[:,:,imageIndex[0]:imageIndex[-1]+1]
        cut3DImageArray = imageArray[:,:,imageIndex[0]:imageIndex[-1]+1]

        return cut3DLabelArray, cut3DImageArray, imageIndex, sliceNumber
    
    if axis=="coronal":
        totalNumber = len(labelArray[0,:,:0])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[:,x,:]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = 15

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[:,imageIndex[0]:imageIndex[-1]+1,:]
        cut3DImageArray = imageArray[:,imageIndex[0]:imageIndex[-1]+1,:]
        
        return cut3DLabelArray, cut3DImageArray,imageIndex
        
    if axis=="sagittal":
        totalNumber = len(imageArray[:,0,0])
        imageIndex = []

    #高さ方向の腎臓、腎臓がんの範囲特定
        for x in range(totalNumber):
            if np.where(labelArray[x,:,:]!=0,True,False).any():
                imageIndex.append(x)

        sliceNumber = 15

        #上下数枚を取り込み
        IndexFirst = imageIndex[0]
        IndexFinal = imageIndex[-1]
        imageIndexFirst = []
        imageIndexFinal = []

        for a in range(sliceNumber):
            if (IndexFirst-a-1)>=0:
                imageIndexFirst.append(IndexFirst-a-1)
            if (IndexFinal+a+1)<totalNumber:
                imageIndexFinal.append(IndexFinal+a+1)

        imageIndexFirst.reverse()
        imageIndex= imageIndexFirst + imageIndex + imageIndexFinal#高さ方向の必要な部分のインデックスの配列

        #高さ方向について必要な部分のみ切り取った3D画像を生成
        cut3DLabelArray = labelArray[imageIndex[0]:imageIndex[-1]+1,:,:]
        cut3DImageArray = imageArray[imageIndex[0]:imageIndex[-1]+1,:,:]
        
        return cut3DLabelArray, cut3DImageArray,imageIndex

def cut_image(imgArray, paddingSize=15, center=None, wh=None, angle=None):#paddingSize(int), center(tuple), wh=(tuple)
    #unsigned int に変換
    area = 0
    if center==None and wh==None and angle==None :
        imgArray = np.array(imgArray,dtype=np.uint8)

        ## 輪郭抽出
        vcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(vcheck)==3:
            _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        else:
            contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # 面積が最大の輪郭を選択する。
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        
        # 外接矩形を取得する。

        center, wh, angle = cv2.minAreaRect(cnt)
    
    if center==None and wh!=None and angle==None :
        imgArray = np.array(imgArray,dtype=np.uint8)

        ## 輪郭抽出
        vcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(vcheck)==3:
            _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        else:
            contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        # 面積が最大の輪郭を選択する。
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        
        # 外接矩形を取得する。

        center, _, angle = cv2.minAreaRect(cnt)
        
    #intに変換
    center = list(center)
    wh = list(wh)
    center = tuple(map(lambda x: int(x), center))
    wh = tuple(map(lambda x: int(x), wh))#wh = (width, height)
    paddedwh =  tuple(map(lambda x: int(x) + paddingSize, wh))#wh = (width, height)
    
    imgwh = (imgArray.shape[1], imgArray.shape[0])#元画像の幅と高さ

    #スケールを指定
    scale = 1.0

    #画像を回転させるための行列を生成
    trans = cv2.getRotationMatrix2D(center, angle , scale)

    #アフィン変換（元画像の行列にさっきの行列をかけて画像を回転）
    rotatedImg = cv2.warpAffine(imgArray, trans, imgwh)
    
    # 切り出す。
    x0 = center[1] - int(paddedwh[1]/2) 
    x1 = center[1] + int(paddedwh[1]/2) 
    y0 = center[0] - int(paddedwh[0]/2)
    y1 = center[0] + int(paddedwh[0]/2)
    
#    if x0<0:
#        x0 = 0
#    if y0<0:
#        y0 = 0
    roi = rotatedImg[x0 : x1 , y0 : y1]
    
    if area==0:
        return roi, center, wh, angle
    
    else:
        ##情報出力
        #print("Area: ",area)
        #print("Center: ",center)
        #print("Original width and height: ",wh)
        #print("Angle: ",angle)
        #print("Padding_size: ",paddingSize)
        #print("Padded width and height: ", paddedwh)
        #print("Original image shape: ",imgArray.shape)
        #print("Rotated image shape: ",rotatedImg.shape)
        #print("ROI image shape: ", roi.shape)
        #print("\n")
    
        return roi, center, wh, angle

def caluculate_area(imgArray):
    area = 0
    #unsigned int に変換
    imgArray = np.array(imgArray,dtype=np.uint8)

    ## 輪郭抽出
    
    vvcheck = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(vvcheck)==3:
        _, contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    else:
        contours, _ = cv2.findContours(imgArray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    
    # 面積が最大の輪郭を選択する。
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    area = cv2.contourArea(cnt)
    
    return area

#For Resampling (maybe mistake)
#def save_image_256(imageArray, image, savePath, is_lab=False):
#    LF = sitk.GetImageFromArray(imageArray)
#    if is_lab:
#        LF = Resampling(LF,(256,256),LF.GetSize(), is_label=True)
#    else:
#        LF = Resampling(LF,(256,256),LF.GetSize())
#
#    
#    sitk.WriteImage(LF, savePath, True)

#For ResamplingInAxis 
def save_image_256(imageArray, image, savePath, is_lab=False):
    LF = sitk.GetImageFromArray(imageArray)
    if is_lab:
        LF = ResamplingInAxis(LF, image, (256,256), is_label=True)
    else:
        LF = ResamplingInAxis(LF, image, (256,256) )

    sitk.WriteImage(LF, savePath, True)

# For moreCancer
def inverse_image(roi, cutKidFragLabel, wh, center, angle, paddingSize=15, clipSize=100):
    blackImg = np.zeros_like(cutKidFragLabel)

    wh = tuple([x+y for x,y in zip(wh,(paddingSize, paddingSize))])
    

    x0 = center[1] - int(wh[1]/2) 
    x1 = center[1] + int(wh[1]/2) 
    y0 = center[0] - int(wh[0]/2)
    y1 = center[0] + int(wh[0]/2)

#    if x0<0:
#        x0 = 0
#    if y0<0:
#        y0 = 0

    blackImg[x0 : x1 , y0 : y1] = roi


    #画像を回転させるための行列を生成
    trans = cv2.getRotationMatrix2D(center, (-1)*angle , 1.0)

    #アフィン変換（元画像の行列にさっきの行列をかけて画像を回転）
    iImg = cv2.warpAffine(blackImg, trans, blackImg.shape[::-1])

    iImg = iImg[clipSize: -clipSize, clipSize: -clipSize] 
    
    return iImg


def saveSliceImage256(imgArray, img, savePath, interpolation):

    from clip3D import Resizing

    argMax = np.argmax(np.array(imgArray.shape))
    savePathList = []
    if argMax == 0:
        axisSize = imgArray.shape[0]
        dummyArray = np.zeros((axisSize, 256, 256))
        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)

        for x in range(axisSize):
            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[x, :, :])

            direction = (0.0, 1.0, -1.0, 0.0)
            origin = img.GetOrigin()[:2]
            spacing = img.GetSpacing()[:2]

            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)

            sitk.WriteImage(resizedImg, saveSlicePath)

    elif argMax == 1:
        axisSize = imgArray.shape[1]
        dummyArray = np.zeros((256, axisSize, 256))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)

        for x in range(axisSize):

            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[:, x, :])

            direction = (0.0, 1.0, -1.0, 0.0)
            origin = (img.GetOrigin()[0], img.GetOrigin()[2])
            spacing = (img.GetSpacing()[0], img.GetSpacing()[2])

            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)

            sitk.WriteImage(resizedImg, saveSlicePath)

    else:
        axisSize = imgArray.shape[2]
        dummyArray = np.zeros((256, 256, axisSize))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)

        for x in range(axisSize):

            saveSlicePath = savePath + str(x).zfill(3) + ".mha"
            savePathList.append(saveSlicePath)
            createParentPath(saveSlicePath)
            resizedImg = sitk.GetImageFromArray(resizedImgArray[:, :, x])

            direction = (0.0, 1.0, -1.0, 0.0)
            origin = img.GetOrigin()[1:]
            spacing = img.GetSpacing()[1:]

            resizedImg.SetDirection(direction)
            resizedImg.SetOrigin(origin)
            resizedImg.SetSpacing(spacing)

            sitk.WriteImage(resizedImg, saveSlicePath)

    return savePathList

### You should revise Resizing.
### This name is not appropriate.
def sliceImage(imgArray, interpolation):

    from clip3D import Resizing

    argMax = np.argmax(np.array(imgArray.shape))
    
    if argMax == 0:
        axisSize = imgArray.shape[0]
        dummyArray = np.zeros((axisSize, 256, 256))
        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)


    elif argMax == 1:
        axisSize = imgArray.shape[1]
        dummyArray = np.zeros((256, axisSize, 256))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)


    else:
        axisSize = imgArray.shape[2]
        dummyArray = np.zeros((256, 256, axisSize))

        resizedImgArray = Resizing(imgArray, dummyArray, interpolation)

    
    return resizedImgArray, argMax

