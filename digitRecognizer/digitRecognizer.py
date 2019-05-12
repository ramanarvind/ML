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
    def __init__(self, algo: str ="knn"):
        self.algo= algo
        if(self.algo == "knn"):
            self.knn = opencv.ml.KNearest_create()

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
