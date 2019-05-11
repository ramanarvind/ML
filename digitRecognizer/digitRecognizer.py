# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:23:27 2019

@author: araman
"""

"""
Check this link for a quick introduction on K-NN 
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_knn/py_knn_understanding/py_knn_understanding.html#knn-understanding

"""
import numpy as np
import cv2 as opencv

class digitRecognizer:
    def __init__(self, algo: str ="knn", relativePath = ""):
        self.algo= algo
        if(self.algo == "knn"):
            self.knn = opencv.ml.KNearest_create()
            fpTrain = open(relativePath+"./train/train-images.idx3-ubyte", "rb")
            magicNum = int.from_bytes(fpTrain.read(4), byteorder='big')
            assert(magicNum == 2051)
            numTrainImgs = int.from_bytes(fpTrain.read(4), byteorder='big')
            numTrainRows = int.from_bytes(fpTrain.read(4), byteorder='big')
            numTrainCols = int.from_bytes(fpTrain.read(4), byteorder='big')
            
            fpTrainLabel = open(relativePath+"./train/train-labels.idx1-ubyte", "rb")
            magicNum = int.from_bytes(fpTrainLabel.read(4), byteorder='big')
            assert(magicNum == 2049)
            numTrainLabels = int.from_bytes(fpTrainLabel.read(4), byteorder='big')
            
            trainVecs = np.frombuffer(fpTrain.read(numTrainImgs*numTrainRows*numTrainCols), dtype='ubyte')
            trainVecs = opencv.adaptiveThreshold(trainVecs, 255, opencv.ADAPTIVE_THRESH_MEAN_C, opencv.THRESH_BINARY_INV, 9, 2);
            trainVecs = np.array(trainVecs).astype(np.float32)
            trainVecs = trainVecs.reshape(numTrainImgs, numTrainRows*numTrainCols)
            
            trainLabels = np.frombuffer(fpTrainLabel.read(numTrainLabels), dtype='ubyte')
            trainLabels = np.array(trainLabels).astype(np.float32)
            trainLabels = trainLabels.reshape(numTrainLabels, 1)
            self.train(trainVecs, trainLabels)


    def knn_train(self, trainVecs, trainLabels):
        self.knn.train(trainVecs, opencv.ml.ROW_SAMPLE, trainLabels)
        
    def knn_test(self, testVecs):
        ret, result, neighbours, dist = self.knn.findNearest(testVecs, k=5)
        return(ret, result, neighbours, dist)
                    
    # Container for a K-NN based digit recognizer
    def train(self, trainVecs, trainLabels):
        if(self.algo == "knn"):
            self.knn_train(trainVecs, trainLabels)

    def test(self, testVecs):
        if(self.algo == "knn"):
            ret, result, neighbours, dist = self.knn_test(testVecs)
            return(ret, result, neighbours, dist)
