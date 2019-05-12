# -*- coding: utf-8 -*-
"""
Created on Thu May  9 04:19:14 2019

@author: araman
"""
import sudoku

"""
sudokuInst = sudoku.sudoku(hCellDim=3, vCellDim=2)
board = [[0, 0, 0, 0, 0, 0], 
         [0, 0, 0, 5, 0, 1], 
         [2, 0, 0, 0, 0, 6], 
         [6, 0, 0, 0, 0, 3],
         [4, 0, 5, 3, 0, 0],
         [0, 0, 0, 0, 0, 0]]
"""

sudokuInst = sudoku.sudoku(hCellDim=3, vCellDim=3)
"""
board = [[6, 0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 4, 0, 0, 0],
         [0, 7, 1, 0, 6, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 8, 0, 6],
         [0, 5, 9, 0, 0, 0, 0, 0, 4],
         [0, 2, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 8, 6, 0, 7, 5, 3, 0],
         [9, 0, 0, 4, 0, 0, 0, 0, 2],
         [3, 0, 0, 0, 0, 0, 0, 0, 9]]
"""
board = [[4, 0, 1, 2, 9, 0, 0, 7, 5],
         [2, 0, 0, 3, 0, 0, 8, 0, 0],
         [0, 7, 0, 0, 8, 0, 0, 0, 6],
         [0, 0, 0, 1, 0, 3, 0, 6, 2],
         [1, 0, 5, 0, 0, 0, 4, 0, 3],
         [7, 3, 0, 6, 0, 8, 0, 0, 0],
         [6, 0, 0, 0, 2, 0, 0, 3, 0],
         [0, 0, 7, 0, 0, 1, 0, 0, 4],
         [8, 9, 0, 0, 6, 5, 1, 0, 7]]

if(sudokuInst.solveSudoku(board)):
    print("Sodoku solved")
    for row in board:
        print(row)
else:
    print("Sodoku was not solved")