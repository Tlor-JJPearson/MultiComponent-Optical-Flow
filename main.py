#import matplotlib.pyplot as plt #unused yet
import numpy as np
import gc
import cv2 as cv
#import skimage.transform
#import time

#from functools import partial
from optical_flow import flow_iterative

counter = 0 #initialized for frame counting
#byteCounter = 0 #unused (unless for debug)

#TAKES IMAGE, FLOW AND RETURNS NEXT IMAGE BASED ON THEM
def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res

#CONVERTS AN ARRAY OF BYTES AND RETURNS IT AS ARRAY OF FLOATS
def byteToFloat(array, w:int, h:int):
     x = 0
     y = 0
     newArray = np.zeros((w, h), dtype = float)
     while (y < h):
        while (x < w):
           if (len(array[x,y]) == 0): array[x,y] = '\x01'
           newArray[x,y] = float(ord(array[x,y]))
           x = x+1
        y = y+1
        x = 0
     return newArray

#READS ANOTHER ARRAY AND CONVERTS IT
def nextArray(array, w:int, h:int):
    x = 0
    y = 0
    while (y < h):
        while (x < w):
            array[x,y] = f.read(1)
            x = x+1
        y = y+1
        x = 0
    convertedArray = byteToFloat(array, w, h)
    return convertedArray 
    
   
#source(<width>x<height>).yuv has resolution (width x height)
#please provide correct parameters

width = int(input("Enter video's width: "))
height = int(input("Enter video's height: "))
limit = int(input("Enter number of flow frames: "))

#open the .yuv file:
#ICE_704x576_30_orig_02.yuv
#HARBOUR_704x576_30_orig_01.yuv
#CREW_704x576_30_orig_01.yuv
#ChinaSpeed_1024x768_30_8bit.yuv
#SlideShow_1280x720_20_8bit.yuv
#Kimono1_1920x1080_24_8bit.yuv 
#NebutaFestival_2560x1600_60_8bit.yuv ------------------------
#CITY(176x144)150.yuv
#CREW(352x288)300.yuv
#MOBILE(352x288)300.yuv
#SOCCER(176x144)150.yuv
#SOCCER(352x288)300.yuv (DEFAULT)
#BUS(176x144)75.yuv
#BUS(352x288)150.yuv
#HARBOUR(176x144)150.yuv
#HARBOUR(352x288)300.yuv
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

    #for algorithms to work, the arrays need to be in
    #64-bit (double precision) floating point format in range <0-255>
    #they are also transposed because I made width and height backwards, silly me
    f1 = np.transpose(nextArray(Y, width, height))
    fu1 = np.transpose(nextArray(U, chrWidth, chrHeight))
    fv1 = np.transpose(nextArray(V, chrWidth, chrHeight))

    ###############################################
    #     PARAMETERS FOR FARNEBACK ALGORITHM
    ###############################################

    #certainty is decreased for pixels near the edge
    #of the image, as recommended by Farneback
    c1 = np.minimum(1, 1 / 5 * np.minimum(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
    c1 = np.minimum(
        c1,
        1
        / 5
        * np.minimum(
            f1.shape[0] - 1 - np.arange(f1.shape[0])[:, None],
            f1.shape[1] - 1 - np.arange(f1.shape[1]),
        ),
    )
    c2 = c1
    
    #for U and V images:
    cc1 = np.minimum(1, 1 / 5 * np.minimum(np.arange(fu1.shape[0])[:, None], np.arange(fu1.shape[1])))
    cc1 = np.minimum(
        cc1,
        1
        / 5
        * np.minimum(
            fu1.shape[0] - 1 - np.arange(fu1.shape[0])[:, None],
            fu1.shape[1] - 1 - np.arange(fu1.shape[1]),
        ),
    )
    cc2 = cc1

    sigma = 4.0
    opts = dict(
        #sigma = 4.0 (not working with this template)
        sigma_flow=4.0,
        num_iter=10,
        model="constant",
        #model="eight_param"
        mu=0,
        #mu = None
    )
    #n_pyr = 4 #UNUSED

    #initial flow (zeroes)
    dy = np.zeros((height, width, 2), dtype = float)
    du = np.zeros((chrHeight, chrWidth, 2), dtype = float)
    dv = np.zeros((chrHeight, chrWidth, 2), dtype = float)

    #hsv image matrices
    hsv_mask = np.zeros((height, width, 3), dtype = np.int8)
    hsv_masku = np.zeros((chrHeight, chrWidth, 3), dtype = np.int8)
    hsv_maskv = np.zeros((chrHeight, chrWidth, 3), dtype = np.int8)
    hsv_maskC = np.zeros((height, width, 3), dtype = np.int8)

    #setting saturation to maximum
    hsv_mask[..., 1] = 255
    hsv_masku[..., 1] = 255
    hsv_maskv[..., 1] = 255
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
    #difference = 0.0

    #use the textfile for printed results
    #result = open("YUV\\result.txt", "w")
    #print("Result textfile ready!")
 
    while(True):

        #getting new frames (transposed for widthxheight orientation)
        f2 = np.transpose(nextArray(Y, width, height)) 
        fu2 = np.transpose(nextArray(U, chrWidth, chrHeight))
        fv2 = np.transpose(nextArray(V, chrWidth, chrHeight))

        #GUNNAR FARNEBACK, NO PYRAMIDS
        dy = flow_iterative(f1, f2, sigma, c1, c2, d=dy , **opts)
        du = flow_iterative(fu1, fu2, sigma, cc1, cc2, d=du , **opts)
        dv = flow_iterative(fv1, fv2, sigma, cc1, cc2, d=dv , **opts)
        #switch vectors x,y to y,x
        dy = dy[..., (1, 0)]
        du = du[..., (1, 0)]
        dv = dv[..., (1, 0)]
       
        #METHOD OF COMBINATION : VECTORS'COORDINATES AVERAGING
        #assume the Y's flow
        combined = np.copy(dy)
        #combined2 = dy
        #upscale both U and V flows to Y
        while (j < chrHeight):
            while (i < chrWidth):
                tempU[2*j, 2*i, 0] = du[j,i,0] * 2
                tempU[2*j+1, 2*i, 0] = du[j,i,0] * 2
                tempU[2*j, 2*i+1, 0] = du[j,i,0] * 2
                tempU[2*j+1, 2*i+1, 0] = du[j,i,0] * 2

                tempU[2*j, 2*i, 1] = du[j,i,1] * 2
                tempU[2*j+1, 2*i, 1] = du[j,i,1] * 2
                tempU[2*j, 2*i+1, 1] = du[j,i,1] * 2
                tempU[2*j+1, 2*i+1, 1] = du[j,i,1] * 2

                tempV[2*j, 2*i, 0] = dv[j,i,0] * 2
                tempV[2*j+1, 2*i, 0] = dv[j,i,0] * 2
                tempV[2*j, 2*i+1, 0] = dv[j,i,0] * 2
                tempV[2*j+1, 2*i+1, 0] = dv[j,i,0] * 2

                tempV[2*j, 2*i, 1] = dv[j,i,1] * 2
                tempV[2*j+1, 2*i, 1] = dv[j,i,1] * 2
                tempV[2*j, 2*i+1, 1] = dv[j,i,1] * 2
                tempV[2*j+1, 2*i+1, 1] = dv[j,i,1] * 2
                
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
                #ONLY CHROMA:
                #combined2[j,i,0] = (tempU[j,i,0] + tempV[j,i,0]) / 2
                #combined2[j,i,1] = (tempU[j,i,1] + tempV[j,i,1]) / 2
                i += 1
            j += 1
            i = 0
        j = 0
        i = 0

        #dycomb = np.mean(np.square(dy - combined))
        #print("Difference between flows:")
        #print(dycomb)
        #RECONSTRUCT f2 IMAGE FROM f1 FLOW:
        f2recon = warp_flow(f1.astype(np.uint8), dy.astype(np.float32))
        f2combRecon = warp_flow(f1.astype(np.uint8), combined.astype(np.float32))
        #f2combRecon2 = warp_flow(f1.astype(np.uint8), combined2.astype(np.float32))

        #CALCULATE AVERAGE DIFFERENCE OF IMAGES:
        '''
        while (j < height):
            while (i < width):
                f2reconSUM += abs(f2[j,i] - f2recon[j,i])
                f2combReconSUM += abs(f2[j,i] - f2combRecon[j,i])
                i += 1
            j += 1
            i = 0
        j = 0
        i = 0
        '''
        f2reconSUM = np.mean(abs(f2 - f2recon))
        f2combReconSUM = np.mean(abs(f2 - f2combRecon))
        f2reconFINAL += f2reconSUM
        f2combReconFINAL += f2combReconSUM
        #difference = np.mean(np.square(f2recon - f2combRecon))


        print("Average difference of Farnebacks reconstruction: ")
        print(f2reconSUM)
        #f2reconSUM = 0.0
        print("Average difference of 3D modified reconstruction: ")
        print(f2combReconSUM)
        #f2combReconSUM = 0.0
        #print("MS between reconstructions: ")
        #print(difference)
        #difference = 0.0

        
        
        #COMPUTE AND CONVERT INTO HSV IMAGES:
        # Compute magnitude and angle of 2D vector
        mag, ang = cv.cartToPolar(dy[..., 0], dy[..., 1])
        magu, angu = cv.cartToPolar(du[..., 0], du[..., 1])
        magv, angv = cv.cartToPolar(dv[..., 0], dv[..., 1])
        magC, angC = cv.cartToPolar(combined[..., 0], combined[..., 1])

        # Set image hue value according to the angle of optical flow
        hsv_mask[..., 0] = ang * 180 / np.pi / 2
        hsv_masku[..., 0] = angu * 180 / np.pi / 2
        hsv_maskv[..., 0] = angv * 180 / np.pi / 2
        hsv_maskC[..., 0] = angC * 180 / np.pi / 2

        # Set value as per the normalized magnitude of optical flow
        hsv_mask[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        hsv_masku[..., 2] = cv.normalize(magu, None, 0, 255, cv.NORM_MINMAX)
        hsv_maskv[..., 2] = cv.normalize(magv, None, 0, 255, cv.NORM_MINMAX)
        hsv_maskC[..., 2] = cv.normalize(magC, None, 0, 255, cv.NORM_MINMAX)

        # Convert to rgb
        rgb = cv.cvtColor(hsv_mask.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbu = cv.cvtColor(hsv_masku.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbv = cv.cvtColor(hsv_maskv.astype(np.uint8), cv.COLOR_HSV2BGR)
        rgbC = cv.cvtColor(hsv_maskC.astype(np.uint8), cv.COLOR_HSV2BGR)

        #VISUALIZE RESULTS (press e to close):
        cv.imshow('Y', rgb.astype(np.uint8))
        kk = cv.waitKey(20) & 0xff
        if kk == ord('e'):
            break
        
        cv.imshow('U', rgbu.astype(np.uint8))
        cv.imshow('V', rgbv.astype(np.uint8))
        cv.imshow('Combined', rgbC.astype(np.uint8))
        cv.imshow('Image', f2.astype(np.uint8))
        cv.imshow('Recon1', f2recon.astype(np.uint8))
        cv.imshow('Recon2', f2combRecon.astype(np.uint8))
        #END OF HSV VISUALIZATION
        

        #write results as 2D arrays (2 components of vectors are in front of each other)
        #one line corresponds to one row on flow
        '''
        dyr = np.reshape(dy,(height,width*2))
        dur = np.reshape(du,(chrHeight,chrWidth*2))
        dvr = np.reshape(dv,(chrHeight,chrWidth*2))
        np.savetxt(result, dyr)
        np.savetxt(result, dur)
        np.savetxt(result, dvr)
        
        print(dy)
        print(dy2)
        '''
        gc.collect()

        #number of frames processed
        counter += 1
        print(counter)
        if (counter == limit or limit <= 0):
            break
        
        #10 seconds time to check 10th frame
        #if (counter == 10):
        #    time.sleep(50)

        #next frame setup:
        f1 = f2
        fu1 = fu2
        fv1 = fv2
    
    if counter <= 0: #idiot-proofing
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

    #print("Average  of Farnebacks reconstruction: ")
    #print(f2reconSUM)
    #print("Average of 3D modified reconstruction: ")
    #print(f2combReconSUM) 

    ### UNUSED CODE BELOW!!!! ####

    # calculate optical flow using pyramids
    # note: reversed(...) because we start with the smallest pyramid
        
    '''
        for pyr1, pyr2, c1_, c2_ in reversed(
            list(
                zip(
                    *list(
                        map(
                            partial(skimage.transform.pyramid_gaussian, max_layer=n_pyr),
                            [f1, f2, c1, c2],
                        )
                    )
                )
            )
        ):
            if d is not None:
                # To do: account for shapes not quite matching
                d = skimage.transform.pyramid_expand(d, channel_axis=-1)
                d = d[: pyr1.shape[0], : pyr2.shape[1]] * 2

                d = flow_iterative(pyr1, pyr2, c1=c1_, c2=c2_, d=d, **opts)
                print(d)
                xw = d + np.moveaxis(np.indices(f1.shape), 0, -1)
    '''
        
