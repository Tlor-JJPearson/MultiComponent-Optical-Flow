#import matplotlib.pyplot as plt #unused yet
import numpy as np
import gc
import cv2 as cv
#import skimage.transform #unused
#import time

#from functools import partial
#from optical_flow import flow_iterative

counter = 0 #initialized for frame counting

#TAKES IMAGE, FLOW AND RETURNS NEXT IMAGE BASED ON THEM
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

#CONVERTS AN ARRAY OF BYTES AND RETURNS IT AS ARRAY OF FLOATS (OWN DESIGN)
def byteToInt8(array, w:int, h:int):
     x = 0
     y = 0
     newArray = np.zeros((w, h), dtype = float)
     while (y < h):
        while (x < w):
           if (len(array[x,y]) == 0): array[x,y] = '\x01'
           newArray[x,y] = ord(array[x,y])
           x = x+1
        y = y+1
        x = 0
     return newArray

#READS ANOTHER ARRAY AND CONVERTS IT (OWN DESIGN)
def nextArray(array, w:int, h:int):
    x = 0
    y = 0
    while (y < h):
        while (x < w):
            array[x,y] = f.read(1)
            x = x+1
        y = y+1
        x = 0
    convertedArray = byteToInt8(array, w, h)
    return convertedArray 
    
   
#source(<width>x<height>).yuv has resolution (width x height)
#please provide correct parameters
#print(cv.__version__)

width = int(input("Enter video's width: "))
height = int(input("Enter video's height: "))
limit = int(input("Enter number of flow frames: "))

#open the .yuv file:
#ICE_704x576_30_orig_02.yuv
#ChinaSpeed_1024x768_30_8bit.yuv
#SlideShow_1280x720_20_8bit.yuv
#Kimono1_1920x1080_24_8bit.yuv
#CITY(176x144)150.yuv
#CREW(352x288)300.yuv
#CREW_704x576_30_orig_01.yuv
#MOBILE(352x288)300.yuv
#SOCCER(176x144)150.yuv
#SOCCER(352x288)300.yuv
#BUS(176x144)75.yuv
#BUS(352x288)150.yuv (DEFAULT)
#HARBOUR(176x144)150.yuv
#HARBOUR(352x288)300.yuv
#HARBOUR_704x576_30_orig_01.yuv
with open("YUV_\\BUS(352x288)150.yuv", "rb") as f:
    print("File opened!")

    # get first frame Y (intensity)
    Y = np.zeros((width, height), dtype = bytes)
    #get chroma U and Chroma V (4:2:0)
    chrWidth = int(width/2)
    chrHeight = int(height/2)
    #chroma U
    U = np.zeros((chrWidth, chrHeight), dtype = bytes)
    #chroma V
    V = np.zeros((chrWidth, chrHeight), dtype = bytes)

    #they are transposed because I made width and height backwards, silly me
    f1 = np.transpose(nextArray(Y, width, height))
    fu1 = np.transpose(nextArray(U, chrWidth, chrHeight))
    fv1 = np.transpose(nextArray(V, chrWidth, chrHeight))

    #PARAMETERS FOR OPENCV FARNEBACK (NO PYRAMIDS)
    opts_cv = dict(
        pyr_scale=0.5,
        levels=5,
        winsize=25,
        iterations=10,
        poly_n=25,
        poly_sigma=4.0,
        # flags=0
        flags=cv.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    #initial flow (zeroes)
    dy2 = np.zeros((height, width, 2), dtype = float)
    du2 = np.zeros((chrHeight, chrWidth, 2), dtype = float)
    dv2 = np.zeros((chrHeight, chrWidth, 2), dtype = float)

    #hsv image matrices
    hsv_mask2 = np.zeros((height, width, 3), dtype = np.int8)
    hsv_masku2 = np.zeros((chrHeight, chrWidth, 3), dtype = np.int8)
    hsv_maskv2 = np.zeros((chrHeight, chrWidth, 3), dtype = np.int8)
    hsv_maskC = np.zeros((height, width, 3), dtype = np.int8)
    #hsv_mask2recon = np.zeros((height, width, 3), dtype = np.int8)

    #setting saturation to maximum
    hsv_mask2[..., 1] = 255
    hsv_masku2[..., 1] = 255
    hsv_maskv2[..., 1] = 255
    hsv_maskC[..., 1] = 255

    #average vectors flow matrix
    combined = np.zeros((height, width, 2), dtype = float)
    #combined2 = np.zeros((height, width, 2), dtype = float)
    tempU = np.zeros((height, width, 2), dtype = float)
    tempV = np.zeros((height, width, 2), dtype = float)
    i = 0
    j = 0

    #reconstruction matrix
    f2recon = np.zeros((height, width), dtype = np.uint8)
    f2combRecon = np.zeros((height, width), dtype = np.uint8)
    f2combRecon2 = np.zeros((height, width), dtype = np.uint8)

    f2reconSUM = 0.0
    f2combReconSUM = 0.0
    f2reconFINAL = 0.0
    f2combReconFINAL = 0.0

    #use the textfile for printed results
    #result = open("YUV\\result.txt", "w")
    #print("Result textfile ready!")
 
    while(True):

        #getting new frames (transposed for WidthxHeight orientation)
        f2 = np.transpose(nextArray(Y, width, height)) 
        fu2 = np.transpose(nextArray(U, chrWidth, chrHeight))
        fv2 = np.transpose(nextArray(V, chrWidth, chrHeight))
        
        #GUNNAR fARNERBACK, OPENCV
        dy2 = cv.calcOpticalFlowFarneback(f1, f2, None, **opts_cv)
        du2 = cv.calcOpticalFlowFarneback(fu1, fu2, None, **opts_cv)
        dv2 = cv.calcOpticalFlowFarneback(fv1, fv2, None, **opts_cv)

        #METHOD OF COMBINATION:S VECTORS AVERAGING
        #assume the Y's flow
        combined = np.copy(dy2)
        #combined2 = dy2
        #upscale both U and V flows to Y
        while (j < chrHeight):
            while (i < chrWidth):
                tempU[2*j, 2*i, 0] = du2[j,i,0] * 2
                tempU[2*j+1, 2*i, 0] = du2[j,i,0] * 2
                tempU[2*j, 2*i+1, 0] = du2[j,i,0] * 2
                tempU[2*j+1, 2*i+1, 0] = du2[j,i,0] * 2

                tempU[2*j, 2*i, 1] = du2[j,i,1] * 2
                tempU[2*j+1, 2*i, 1] = du2[j,i,1] * 2
                tempU[2*j, 2*i+1, 1] = du2[j,i,1] * 2
                tempU[2*j+1, 2*i+1, 1] = du2[j,i,1] * 2

                tempV[2*j, 2*i, 0] = dv2[j,i,0] * 2
                tempV[2*j+1, 2*i, 0] = dv2[j,i,0] * 2
                tempV[2*j, 2*i+1, 0] = dv2[j,i,0] * 2
                tempV[2*j+1, 2*i+1, 0] = dv2[j,i,0] * 2

                tempV[2*j, 2*i, 1] = dv2[j,i,1] * 2
                tempV[2*j+1, 2*i, 1] = dv2[j,i,1] * 2
                tempV[2*j, 2*i+1, 1] = dv2[j,i,1] * 2
                tempV[2*j+1, 2*i+1, 1] = dv2[j,i,1] * 2
                
                i += 1
            j += 1
            i = 0
        j = 0
        i = 0
        #average three matrices's values (coordinates of vectors)
        while (j < height):
            while (i < width):
                combined[j,i,0] = (combined[j,i,0] +  tempU[j,i,0] + tempV[j,i,0]) / 3
                combined[j,i,1] = (combined[j,i,1] +  tempU[j,i,1] + tempV[j,i,1]) / 3
                i += 1
            j += 1
            i = 0
        j = 0
        i = 0
        
        #RECONSTRUCT f2 IMAGE FROM f1 FLOW:
        f2recon = warp_flow(f1.astype(np.uint8), dy2.astype(np.float32))
        f2combRecon = warp_flow(f1.astype(np.uint8), combined.astype(np.float32))

        #CALCULATE AVERAGE DIFFERENCE OF IMAGES:
        f2reconSUM = np.mean(abs(f2 - f2recon))
        f2combReconSUM = np.mean(abs(f2 - f2combRecon))
        f2reconFINAL += f2reconSUM
        f2combReconFINAL += f2combReconSUM

        print("Average difference Farnebacks reconstruction: ")
        print(f2reconSUM)
        #f2reconSUM = 0.0
        print("Average difference 3D modified reconstruction: ")
        print(f2combReconSUM)
        #f2combReconSUM = 0.0
        
        #COMPUTE AND CONVERT INTO HSV IMAGES:
        # Compute magnitude and angle of 2D vector
        mag2, ang2 = cv.cartToPolar(dy2[..., 0], dy2[..., 1])
        magu2, angu2 = cv.cartToPolar(du2[..., 0], du2[..., 1])
        magv2, angv2 = cv.cartToPolar(dv2[..., 0], dv2[..., 1])
        magC, angC = cv.cartToPolar(combined[..., 0], combined[..., 1])

        # Set image hue value according to the angle of optical flow
        hsv_mask2[..., 0] = ang2 * 180 / np.pi / 2
        hsv_masku2[..., 0] = angu2 * 180 / np.pi / 2
        hsv_maskv2[..., 0] = angv2 * 180 / np.pi / 2
        hsv_maskC[..., 0] = angC * 180 / np.pi / 2

        # Set value as per the normalized magnitude of optical flow
        hsv_mask2[..., 2] = cv.normalize(mag2, None, 0, 255, cv.NORM_MINMAX)
        hsv_masku2[..., 2] = cv.normalize(magu2, None, 0, 255, cv.NORM_MINMAX)
        hsv_maskv2[..., 2] = cv.normalize(magv2, None, 0, 255, cv.NORM_MINMAX)
        hsv_maskC[..., 2] = cv.normalize(magC, None, 0, 255, cv.NORM_MINMAX)

        # Convert to rgb
        rgb2 = cv.cvtColor(hsv_mask2.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbu2 = cv.cvtColor(hsv_masku2.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbv2 = cv.cvtColor(hsv_maskv2.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbC = cv.cvtColor(hsv_maskC.astype(np.uint8), cv.COLOR_HSV2BGR)
        
        #VISUALIZE RESULTS (press e to close):
        cv.imshow('Y', rgb2.astype(np.uint8))
        kk = cv.waitKey(20) & 0xff
        if kk == ord('e'):
            break
        
        #cv.imshow('Combined', draw_hsv(combined))
        cv.imshow('U', rgbu2.astype(np.uint8))
        cv.imshow('V', rgbv2.astype(np.uint8))
        cv.imshow('Combined', rgbC.astype(np.uint8))
        cv.imshow('Image', f2.astype(np.uint8))
        cv.imshow('Recon1', f2recon.astype(np.uint8))
        cv.imshow('Recon2', f2combRecon.astype(np.uint8))
        #cv.imshow('Recon3', f2combRecon2.astype(np.uint8))
        #HSV VISUALIZATION END
        
        gc.collect()

        #number of frames processed
        counter += 1
        print(counter)
        if (counter == limit or limit <= 0):
            break
        
        #next frame setup:
        f1 = f2
        fu1 = fu2
        fv1 = fv2
    
    if counter <= 0: 
        counter = 1

    #PRINT OVERALL RESULTS OF OBJECTIVE EVALUATION:
    print("-----------------")
    print("Sum af all mean differences of Farnebacks reconstruction: ")
    print(f2reconFINAL)
    print("Sum af all mean differences of 3D Farnebacks reconstruction: ")
    print(f2combReconFINAL)
    print("-----------------")
    print("Average of all mean differences of Farnebacks reconstruction: ")
    print(f2reconFINAL/(counter))
    print("Average of all mean differences of 3D Farnebacks reconstruction: ")
    print(f2combReconFINAL/(counter))

