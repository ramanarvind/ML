# -*- coding: utf-8 -*-
"""
Created on Sat May 11 23:49:00 2019

@author: araman
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:26:37 2019
This file creates a digit recognizer using the dataset provided here 
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

@author: araman
"""


import glob
import numpy as np
import cv2

DIGIT_DIM   = 20
KNN_DIM     = 28

def readFileAndPrepareLabels(dirArray, vecs, labels, op: str="Train"):
    numSamples = 0
    for dirElem in dirArray:
        print("Processing directory", dirElem[0])
        filePattern = dirElem[0]+"*.png"
        fileList = glob.glob(filePattern)
        numFiles = len(fileList)
        
        if(op=="Train"):
            start = 0
            end = int(numFiles*0.5)
        else:
            start = int(numFiles*0.5)
            end = numFiles
            
        for i in range(start, end) :
            file  = fileList[i]
            digit = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            # Scale the digit to a 20x20 size
            #digit = cv2.resize(digit, (DIGIT_DIM,DIGIT_DIM), 0, 0, cv2.INTER_LANCZOS4 )
            # Convert the image to a binary image
            digit = cv2.adaptiveThreshold(digit, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 2);
          
            # Now create a bounding box around the digit
            result = cv2.boundingRect(digit)
           
            if(result[2] > 0) and (result[3] > 0):
                """
                # Scale while preserving aspect ratio
                xScale = DIGIT_DIM / result[2]
                yScale = DIGIT_DIM / result[3]
                scaleFactor = min(xScale, yScale)
                xSize = int(round(scaleFactor * result[2])/2)*2
                ySize = int(round(scaleFactor * result[3])/2)*2
                
                # Scale the digit to a 20x20 size
                digit = cv2.resize(digit[result[1]:result[1]+result[3],result[0]:result[0]+result[2]], (xSize,ySize), 0, 0, cv2.INTER_LANCZOS4)
                # Pad the image with black to make it KNN_DIM x KNN_DIM size                
                digit = cv2.copyMakeBorder(digit, int((KNN_DIM-ySize)/2), int((KNN_DIM-ySize)/2), 
                                           int((KNN_DIM-xSize)/2), int((KNN_DIM-xSize)/2), cv2.BORDER_CONSTANT, 0)
                """
                digit = cv2.resize(digit[result[1]:result[1]+result[3],result[0]:result[0]+result[2]], (DIGIT_DIM,DIGIT_DIM), 0, 0, cv2.INTER_LANCZOS4)
                # Pad the image with black to make it KNN_DIM x KNN_DIM size                
                digit = cv2.copyMakeBorder(digit, int((KNN_DIM-DIGIT_DIM)/2), int((KNN_DIM-DIGIT_DIM)/2), 
                                           int((KNN_DIM-DIGIT_DIM)/2), int((KNN_DIM-DIGIT_DIM)/2), cv2.BORDER_CONSTANT, 0)
                vecs = np.append(vecs, digit)
                labels = np.append(labels, dirElem[1])
                numSamples += 1
            
            """
            cv2.namedWindow('Digit',cv2.WINDOW_NORMAL)
            cv2.imshow('Digit', digit) 
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

    return(numSamples, vecs, labels)

def prepareChars74KVecsLabels(dirArray):
    trainVecs = np.zeros(KNN_DIM*KNN_DIM, dtype='ubyte')
    trainLabels = np.zeros(1, dtype='ubyte')
    numTrainSamples, trainVecs, trainLabels = readFileAndPrepareLabels(dirArray, trainVecs, trainLabels, "Train")
    
    # Reshape both the vectors and the labels
    trainVecs = trainVecs[KNN_DIM*KNN_DIM:].reshape(numTrainSamples,KNN_DIM*KNN_DIM)
    trainVecs = np.array(trainVecs).astype(np.float32)
    trainLabels = trainLabels[1:].reshape(numTrainSamples, 1)
    return(trainVecs, trainLabels)
    