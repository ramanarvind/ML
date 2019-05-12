# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:26:37 2019

@author: araman
"""

import numpy as np
import struct
import cv2
import sys
# Import user defined modules
import sudoku as sudokuSolver
relativePathToDR = "../digitRecognizer"
sys.path.insert(0, relativePathToDR)
import digitRecognizer as DR
from Chars74KUtils import prepareChars74KVecsLabels


"""
Identifies the top, bottom, left and right ourter borders of a sudoku puzzle or 
cell within it and bypasses them
"""
def identifyAndBypassBorder(sudoku, plane: str="cell"):
    hsum = sudoku.sum(axis=1)
    top    = -1
    bottom = -1
    threshold = sudoku.shape[1]*255*0.70
    
    if(plane == "cell"):
        searchFactor = 5
    else:
        searchFactor = 3
    
    #First identify the top and the bottom border
    for y in range(0,int(sudoku.shape[0]/searchFactor)):
        if(hsum[y] >= threshold) and (top == -1):
            top = y
        if((hsum[sudoku.shape[0]-y-1]) >= threshold) and (bottom == -1):
            bottom = sudoku.shape[0]-y-1
    
    # Now bypass the top border
    if(top != -1):
        searchStart = top
        borderBreak = 0 
        for y in range(searchStart, max(searchStart*2, sudoku.shape[0], 32)):
            if(hsum[y] >= threshold) and (borderBreak < 16):
                top = y+1
                borderBreak = 0
            else:
                borderBreak += 1
    else:
        top = 0

    # Now bypass the bottom border
    if(bottom != -1):
        searchStart = bottom
        borderBreak = 0
        for y in range(searchStart,min(sudoku.shape[0]-((sudoku.shape[0] - searchStart)*2), sudoku.shape[0]-32),-1):
            if(hsum[y] >= threshold) and (borderBreak < 16):
                bottom = y-1
                borderBreak = 0
            else:
                borderBreak += 1
    else:
        bottom = sudoku.shape[0]
        
    # Now identify the left and the right border        
    vsum = sudoku.sum(axis=0)
    left  = -1
    right = -1
    threshold = sudoku.shape[0]*255*0.70
    for x in range(0,int(sudoku.shape[1]/searchFactor)):
        if(vsum[x] >= threshold) and (left == -1):
            left = x
        if((vsum[sudoku.shape[1]-x-1]) >= threshold) and (right == -1):
            right = sudoku.shape[1]-x-1
        if(left != -1) and (right != -1):
            break
    
    # As before, bypass the left border
    if(left != -1):
        searchStart = left
        borderBreak = 0
        for x in range(searchStart,max(searchStart*2, sudoku.shape[1], 32)):
            if(vsum[x] >= threshold) and (borderBreak < 16):
                left = x+1
                borderBreak = 0
            else:
                borderBreak += 1
    else:
        left = 0
        
    # and then bypass the right border
    if(right != -1):
        searchStart = right
        borderBreak = 0
        for x in range(searchStart,min(sudoku.shape[1]-((sudoku.shape[1] - searchStart)*2),sudoku.shape[1]-32),-1):
            if(vsum[x] >= threshold) and (borderBreak < 16):
                right = x-1
                borderBreak = 0
            else:
                borderBreak += 1
    else:
        right = sudoku.shape[1]
    
    return(top, bottom, left, right)

"""
main()
Test code starts here
"""

# Path to the dataset provided for printed digits
# http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
Chars74KDataSetPath = [['../digitRecognizer/train/EnglishFnt/Fnt/Sample001/', 0],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample002/', 1],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample003/', 2],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample004/', 3],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample005/', 4],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample006/', 5],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample007/', 6],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample008/', 7],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample009/', 8],
            ['../digitRecognizer/train/EnglishFnt/Fnt/Sample010/', 9]]

SUDOKU_IMG  = "./test/sudoku-1.jpg"
ROW_DIM     = 9
COL_DIM     = 9
DIGIT_DIM   = 20
KNN_DIM     = 28

DUMP_KNN_INPUT = 0
numKnnInputs = 0
KNN_DIGITS_DAT = "./test/cells/sudokucells-2.dat"
KNN_DIGITS_LBL =  "./test/cells/sudokucells-2.lbl"
if(DUMP_KNN_INPUT):
    fdDumpKnnInput = open(KNN_DIGITS_DAT,'wb')
    fdDumpKnnInput.write(struct.pack('i',2051))
    fdDumpKnnInput.write(struct.pack('i',numKnnInputs))
    fdDumpKnnInput.write(struct.pack('i',KNN_DIM))
    fdDumpKnnInput.write(struct.pack('i',KNN_DIM))

# Create a board of dimension ROW_DIM x COL_DIM
board = np.zeros((ROW_DIM, COL_DIM), np.int8)

# Read the input sudoku image
sudokuImg = cv2.imread(SUDOKU_IMG, cv2.IMREAD_UNCHANGED)
# Convert the image to grayscale and with a single channel
sudokuImg = cv2.cvtColor(sudokuImg, cv2.COLOR_BGR2GRAY)
# Blur the image to remove any noise from it
sudokuImg = cv2.GaussianBlur(sudokuImg, (5,5), cv2.BORDER_DEFAULT)
# Convert the image to a binary image
sudokuImg = cv2.adaptiveThreshold(sudokuImg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 2);
"""
# Dilate the image to join broken lines
dilationKernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
sudokuImg = cv2.dilate(sudokuImg, dilationKernel)
"""

"""
# Show the image processed sudoku image
cv2.namedWindow('Sudoku',cv2.WINDOW_NORMAL)
cv2.imshow('Sudoku', sudokuImg) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
Find the top and the bottom boundaries. The boundaries are expected to be 
white. We presume that more than 75% of the picture, both in the horizontal 
and the vertical direction consists of the sudoku puzzle. Hence as soon as we 
find more number of white pixels crosses this threshold we presume we have 
found the puzzle boundary.
"""
# Identify the boundary of the sudoku puzzle
top, bottom, left, right = identifyAndBypassBorder(sudokuImg, plane = "puzzle")
# Crop the image to the identified boundaries
sudokuImg = sudokuImg[top:bottom,left:right]
"""
# Show the cropped sudoku image
cv2.namedWindow('Sudoku',cv2.WINDOW_NORMAL)
cv2.imshow('Sudoku', sudokuImg) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Train the digit recognizer model
trainVecs, trainLabels = prepareChars74KVecsLabels(Chars74KDataSetPath)
# Create an instance of the digit recognizer class and train it
dr = DR.digitRecognizer()
dr.train(trainVecs, trainLabels)

# Now split the image into equal parts horizontally and vertically depending
# upon the grid dimensions (ROW_DIM, COL_DIM) and recognize digits
cell_hdim = sudokuImg.shape[1] / COL_DIM
cell_vdim = sudokuImg.shape[0] / ROW_DIM
for y in range(0,ROW_DIM):
    for x in range(0,COL_DIM):
        top = int(y*cell_vdim)
        bottom = int((y+1)*cell_vdim)
        left = int(x*cell_hdim)
        right = int((x+1)*cell_hdim)
        
        # Preprocess the cell to extract just the digit
        cell = sudokuImg[top:bottom,left:right]
        top, bottom, left, right = identifyAndBypassBorder(cell)
        cell = cell[top:bottom,left:right]

        cell = cv2.GaussianBlur(cell, (5,5), cv2.BORDER_DEFAULT)
        cell[cell < 50] = 0       
        result = cv2.boundingRect(cell)
        
        if(result[2] > 5) and (result[3] > 5):                      
            # Scale the digit to a 20x20 size
            cell = cv2.resize(cell[result[1]:result[1]+result[3],result[0]:result[0]+result[2]], (DIGIT_DIM,DIGIT_DIM), 0, 0, cv2.INTER_LANCZOS4 )

            # Binarize the image again
            #result, cell = cv2.threshold(cell, 50, 255, cv2.THRESH_BINARY)            
            
            # Pad the image with black to make it KNN_DIM x KNN_DIM size                
            cell = cv2.copyMakeBorder(cell, 4, 4, 4, 4, cv2.BORDER_CONSTANT, 0)
            
            # Convert the cell to float 32 type for digit recognizer           
            cellf32 = np.array(cell).astype(np.float32).reshape(1, KNN_DIM*KNN_DIM)
            ret, result, neighbours, dist = dr.test(cellf32)
            #print(y, x, result[0])
            board[y][x] = int(result[0])

            # Dump cells to a file to serve as test vectors for digit recognizer
            if(DUMP_KNN_INPUT == 1):
                numKnnInputs += 1
                cell.astype('int8').tofile(fdDumpKnnInput)        
        
        """
        # Now using center-of-mass place it in a 28x28 dimension space as done 
        # in the MINST dataset
        contours, hierarchy = cv2.findContours(cell, cv2.RETR_EXTERNAL, 2)
        if(len(contours) > 0):
            M = cv2.moments(contours[0])
            if (M['m00'] > 0):
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])            
                #cv2.circle(cell, (cx, cy), 3, (127, 127, 127), -1)
                padTop = int((KNN_DIM/2)-cy)
                padBottom = int(KNN_DIM-((KNN_DIM/2-cy)+DIGIT_DIM))
                padLeft = int((KNN_DIM/2)-cx)
                padRight = int(KNN_DIM-((KNN_DIM/2-cx)+DIGIT_DIM))
                
                if(padTop < 0):
                    padBottom += padTop
                    padTop = 0
                elif(padBottom < 0):
                    padTop += padBottom
                    padBottom = 0

                if(padLeft < 0):
                    padRight += padLeft
                    padLeft = 0
                elif(padRight < 0):
                    padLeft += padRight
                    padRight = 0

                # Pad the image with black to make it KNN_DIM x KNN_DIM size                
                cell = cv2.copyMakeBorder(cell, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, 0)
                
                
                cv2.namedWindow('Sudoku',cv2.WINDOW_NORMAL)
                cv2.imshow('Sudoku', cell) 
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                
                # Dump cells to a file to serve as test vectors for digit recognizer
                if(DUMP_KNN_INPUT == 1):
                    numKnnInputs += 1
                    cell.astype('int8').tofile(fdDumpKnnInput)
                
                # Convert the cell to float 32 type for digit recognizer           
                cellf32 = np.array(cell).astype(np.float32).reshape(1, KNN_DIM*KNN_DIM)
                ret, result, neighbours, dist = dr.test(cellf32)
                board[y][x] = int(result[0])
                
        """
print("Printing the board")
for row in board:
    print(row)
sudoku = sudokuSolver.sudoku(hCellDim=3, vCellDim=3)
if(sudoku.solveSudoku(board)):
    print("Sodoku solved")
    for row in board:
        print(row)
else:
    print("Sodoku was not solved")

if(DUMP_KNN_INPUT == 1):
    fdDumpKnnInput.seek(4)
    fdDumpKnnInput.write(struct.pack('i',numKnnInputs))
    fdDumpKnnInput.close()
    # Sudoku - 1
    # digits = [4, 1, 2, 9, 7, 5, 2, 3, 8, 7, 8, 6, 1, 3, 6, 2, 1, 5, 4, 3, 7, 3, 6, 8, 6, 2, 3, 7, 1, 4, 8, 9, 6, 5, 1, 7]
    # Sudoku - 2
    # digits = [6, 2, 7, 1, 8, 3, 7, 1, 7, 8, 4, 6, 9, 2, 2, 8, 3, 1, 6, 6, 1, 3]
    # Sudoku - 3
    digits = [6, 1, 4, 7, 1, 6, 8, 6, 5, 9, 4, 2, 8, 6, 7, 5, 3, 9, 4, 2, 3, 9]
    # Write labels
    fdDumpKnnInput = open(KNN_DIGITS_LBL, 'wb')
    fdDumpKnnInput.write(struct.pack('i',2049))
    fdDumpKnnInput.write(struct.pack('i',numKnnInputs))
    fdDumpKnnInput.write(bytearray(digits))
    fdDumpKnnInput.close()