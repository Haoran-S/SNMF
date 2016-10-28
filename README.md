# SNMF
C code for our sBSUM algorithm (parallel version), to reproduce our works on SNMF research.

Simply configure and run our code, you will get the result for parallel sBSUM algorithm shown in section 5, fig. 5. To get results for other figures, slightly modification may apply.

To run our code, MPI need to be installed first. 

Our C code need following data, which need to be generated first.

problem dimension - 'dim.txt', ex. 16242	 10

problem matrix - 'dataM%d.txt', ex. y separate 16242 * (16242/y) matrix, where y denotes the number of processors we use.

initial value - 'dataX.txt', ex. a random generated 16242 * 10 matrix

References:

[1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
   "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
   arXiv preprint arXiv:1607.03092 (2016).

version 1.0 -- April/2016

Written by Haoran Sun (hrsun AT iastate.edu)
