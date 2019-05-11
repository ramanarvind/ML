# -*- coding: utf-8 -*-
"""
Created on Mon May  6 09:26:37 2019

@author: araman
"""

import numpy as np
import cv2
import sys
# Import user defined modules
relativePathToDR = "../digitRecognizer"
sys.path.insert(0, relativePathToDR)
import digitRecognizer as DR

"""
Identifies the top, bottom, left and right ourter borders of the image and 
bypass them
"""
def centerOfMassAndPadToDigDim(cell):
    return
    

"""
Identifies the top, bottom, left and right ourter borders of the image and 
bypass them
"""
def identifyAndBypassBorder(sudoku):
    hsum = sudoku.sum(axis=1)
    top    = -1
    bottom = -1
    threshold = sudoku.shape[1]*255*0.8
    
    #First identify the top and the bottom border
    for y in range(0,int(sudoku.shape[0]/5)):
        if(hsum[y] > threshold) and (top == -1):
            top = y
        if((hsum[sudoku.shape[0]-y-1]) > threshold) and (bottom == -1):
            bottom = sudoku.shape[0]-y-1
    
    # Now bypass the top border
    if(top != -1):
        searchStart = top
        borderBreak = 0 
        for y in range(searchStart, max(searchStart*2, sudoku.shape[0], 32)):
            if(hsum[y] > threshold) and (borderBreak < 16):
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
            if(hsum[y] > threshold) and (borderBreak < 16):
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
    threshold = sudoku.shape[0]*255*0.8
    for x in range(0,int(sudoku.shape[1]/5)):
        if(vsum[x] > threshold) and (left == -1):
            left = x
        if((vsum[sudoku.shape[1]-x-1]) > threshold) and (right == -1):
            right = sudoku.shape[1]-x-1
        if(left != -1) and (right != -1):
            break
    
    # As before, bypass the left border
    if(left != -1):
        searchStart = left
        borderBreak = 0
        for x in range(searchStart,max(searchStart*2, sudoku.shape[1], 32)):
            if(vsum[x] > threshold) and (borderBreak < 16):
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
            if(vsum[x] > threshold) and (borderBreak < 16):
                right = x-1
                borderBreak = 0
            else:
                borderBreak += 1
    else:
        right = sudoku.shape[1]
    
    return(top, bottom, left, right)

sudoku = cv2.imread('./test/sudoku-1.jpg', cv2.IMREAD_UNCHANGED)
# Convert the image to grayscale and with a single channel
sudoku = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
# Blur the image to remove any noise from it
sudoku = cv2.GaussianBlur(sudoku, (5,5), cv2.BORDER_DEFAULT)
# Convert the image to a binary image
sudoku = cv2.adaptiveThreshold(sudoku, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 2);
# Dilate the image to join broken lines
dilationKernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
sudoku = cv2.dilate(sudoku, dilationKernel)

# Identify the boundary of the sudoku puzzle
"""
Find the top and the bottom boundaries. The boundaries are expected to be 
white. We presume that more than 75% of the picture, both in the horizontal 
and the vertical direction consists of the sudoku puzzle. Hence as soon as we 
find more number of white pixels crosses this threshold we presume we have 
found the puzzle boundary.
"""
top, bottom, left, right = identifyAndBypassBorder(sudoku)

# Crop the image to the identified boundaries
sudoku = sudoku[top:bottom,left:right]
"""
cv2.namedWindow('Sudoku',cv2.WINDOW_NORMAL)
cv2.imshow('Sudoku', sudoku) 
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Now split the image into equal parts horizontally and vertically depending
# upon the grid dimensions (ROW_DIM, COL_DIM) and recognize digits
ROW_DIM     = 9
COL_DIM     = 9
DIGIT_DIM   = 20
KNN_DIM     = 28

board = [[0, 0, 0, 0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0]]

# Train the digit recognizer model
dr = DR.digitRecognizer(relativePath=relativePathToDR)

cell_hdim = sudoku.shape[1] / COL_DIM
cell_vdim = sudoku.shape[0] / ROW_DIM
for y in range(0,1):
    for x in range(7,8):
        top = int(y*cell_vdim)
        bottom = int((y+1)*cell_vdim)
        left = int(x*cell_hdim)
        right = int((x+1)*cell_hdim)
        
        # Preprocess the cell to extract just the digit
        cell = sudoku[top:bottom,left:right]
        top, bottom, left, right = identifyAndBypassBorder(cell)
        cell = cell[top:bottom,left:right]
        
        # Scale the digit to a 20x20 size
        cell = cv2.resize(cell, (DIGIT_DIM,DIGIT_DIM), 0, 0, cv2.INTER_AREA)
        result, cell = cv2.threshold(cell, 50, 255, cv2.THRESH_BINARY)
        
        # Now using center-of-mass place it in a 28x28 dimension space as done 
        # in the MINST dataset
        contours, hierarchy = cv2.findContours(cell, cv2.RETR_EXTERNAL, 2)
        if(len(contours) > 0):
            M = cv2.moments(contours[0])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])            
            #cv2.circle(cell, (cx, cy), 3, (127, 127, 127), -1)
            padTop = int((KNN_DIM/2)-cy)
            padBottom = int(KNN_DIM-((KNN_DIM/2-cy)+DIGIT_DIM))
            padLeft = int((KNN_DIM/2)-cx)
            padRight = int(KNN_DIM-((KNN_DIM/2-cx)+DIGIT_DIM))
            cell = cv2.copyMakeBorder(cell, padTop, padBottom, padLeft, padRight, cv2.BORDER_CONSTANT, 0)
            
            contours2, hierarchy2 = cv2.findContours(cell, cv2.RETR_EXTERNAL, 2)
            M = cv2.moments(contours2[0])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])            
            print(cx,"x",cy)
            
            # Convert the cell to float 32 type for digit recognizer           
            cellf32 = np.array(cell).astype(np.float32).reshape(1, KNN_DIM*KNN_DIM)
            ret, result, neighbours, dist = dr.test(cellf32)
            board[y][x] = int(result[0])
            
        """
        # Create a window to resize the image to the size of the screen
        #cv2.namedWindow('Sudoku',cv2.WINDOW_NORMAL)
        cv2.imshow('Sudoku', cell) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
print(board)
    

