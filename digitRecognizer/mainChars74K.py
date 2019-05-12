# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:26:37 2019
This file creates a digit recognizer using the dataset provided here 
http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

@author: araman
"""


import numpy as np
import digitRecognizer as DR
from Chars74KUtils import prepareChars74KVecsLabels, readFileAndPrepareLabels

DIGIT_DIM   = 20
KNN_DIM     = 28

dirs = [['./train/EnglishFnt/Fnt/Sample001/', 0],
        ['./train/EnglishFnt/Fnt/Sample002/', 1],
        ['./train/EnglishFnt/Fnt/Sample003/', 2],
        ['./train/EnglishFnt/Fnt/Sample004/', 3],
        ['./train/EnglishFnt/Fnt/Sample005/', 4],
        ['./train/EnglishFnt/Fnt/Sample006/', 5],
        ['./train/EnglishFnt/Fnt/Sample007/', 6],
        ['./train/EnglishFnt/Fnt/Sample008/', 7],
        ['./train/EnglishFnt/Fnt/Sample009/', 8],
        ['./train/EnglishFnt/Fnt/Sample010/', 9]]

trainVecs, trainLabels = prepareChars74KVecsLabels(dirs)
# Create an instance of the digit recognizer class and train it
dr = DR.digitRecognizer()
dr.train(trainVecs, trainLabels)

# Test the model
print("Testing the model against the remaining dataset entries")
for dirElem in dirs:
    testVecs = np.zeros(KNN_DIM*KNN_DIM, dtype='ubyte')
    testLabels = np.zeros(1, dtype='ubyte')
    numTestSamples, testVecs, testLabels = readFileAndPrepareLabels([dirElem], testVecs, testLabels, "Test")
    
    # Reshape both the vectors and the labels
    testVecs = testVecs[KNN_DIM*KNN_DIM:].reshape(numTestSamples,KNN_DIM*KNN_DIM)
    testVecs = np.array(testVecs).astype(np.float32)
    testLabels = testLabels[1:].reshape(numTestSamples, 1)

    ret, result, neighbours, dist = dr.test(testVecs)
    
    matches = result ==  testLabels
    correct = np.count_nonzero(matches)
    accuracy = correct * 100.0 / result.size
    print("Accuracy for digit ", dirElem[1], " = ", accuracy)


# Test the model against the sudoku digits
print("")
print("")
print("Testing the model against the images stored from sudoku")
fpTest = open("../sudokuSolver/test/cells/sudokucells-3.dat", "rb")
magicNum = int.from_bytes(fpTest.read(4), byteorder='little')
assert(magicNum == 2051)
numTestImgs = int.from_bytes(fpTest.read(4), byteorder='little')
numTestRows = int.from_bytes(fpTest.read(4), byteorder='little')
numTestCols = int.from_bytes(fpTest.read(4), byteorder='little')

# Read the header information test labels
fpTestLabel = open("../sudokuSolver/test/cells/sudokucells-3.lbl", "rb")
magicNum = int.from_bytes(fpTestLabel.read(4), byteorder='little')
assert(magicNum == 2049)
numTestLabels = int.from_bytes(fpTestLabel.read(4), byteorder='little')

print("Reading test images...")
testVecs = np.frombuffer(fpTest.read(numTestImgs*numTestRows*numTestCols), dtype='ubyte')
#testVecs = cv2.adaptiveThreshold(testVecs, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2);
testVecs = np.array(testVecs).astype(np.float32)
testVecs = testVecs.reshape(numTestImgs, numTestRows*numTestCols)

print("Reading test labels...")
testLabels = np.frombuffer(fpTestLabel.read(numTestLabels), dtype='ubyte')
testLabels = np.array(testLabels).astype(np.float32)
testLabels = testLabels.reshape(numTestLabels, 1)

print("Finding the nearest neighbours...")
ret, result, neighbours, dist = dr.test(testVecs)

matches = result ==  testLabels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / result.size
print("Accuracy = ", accuracy)
