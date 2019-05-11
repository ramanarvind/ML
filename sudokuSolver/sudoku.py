# -*- coding: utf-8 -*-
"""
Created on Thu May  9 04:19:14 2019

@author: araman
"""

class sudoku:
    def __init__(self, hCellDim: int =3, vCellDim: int =3):
        self.hCellDim = hCellDim
        self.vCellDim = vCellDim
        
    def isValid(self, board, row, col, val):
        # Check that val is not there anywhere else in that same row
        if val in board[row]:
            return False
        # Check that val is not there anywhere else in that same col
        for i in range(0, len(board)):
            if(board[i][col] == val):
                return False
        # Check that val is not there anywhere else in that same box
        rowStart = int(row/self.vCellDim)*self.vCellDim
        colStart = int(col/self.hCellDim)*self.hCellDim
        for i in range(rowStart, rowStart+self.vCellDim):
            for j in range(colStart, colStart+self.hCellDim):
                if(board[i][j] == val):
                    return False
        return True
        
    def solveSodoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        for row in range(0,len(board)):
            for col in range(0,len(board[0])):
                # If the cell is empty
                if board[row][col] == 0:
                    # Check if value i can be placed there
                    for val in range(1,self.hCellDim*self.vCellDim+1):
                        if(self.isValid(board, row, col, val)):
                            board[row][col] = val
                            if(self.solveSodoku(board)):
                                return True
                            else:
                                board[row][col] = 0

                    if(board[row][col] == 0):
                        return False
        return True


sudoku = sudoku(hCellDim=3, vCellDim=2)
board = [[0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 5, 0, 1], 
         [2, 0, 0, 0, 0, 6], 
         [6, 0, 0, 0, 0, 3],
         [4, 0, 5, 3, 0, 0],
         [0, 0, 0, 0, 0, 0]]
if(sudoku.solveSodoku(board)):
    print("Sodoku solved")
    for row in board:
        print(row)
else:
    print("Sodoku was not solved")