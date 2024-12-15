"""
=========================================
Author: Qizhi He
Computational Mathematics Group
Advanced Computing, Mathematics and Data Division
Pacific Northwest National Laboratory 
Email: qzhe.ucsd@gmail.com; qizhi.he@pnnl.gov
https://sites.google.com/view/qzhe
=========================================
"""
import numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)

# Generate coordinates vector for uniform grids over a 2D rectangles. Order starts from left-bottom, row-wise, to right-up
def rectspace3D(lb,ub,n1,n2,n3):
	x1 = np.linspace(lb[0],ub[0],n1)
	x2 = np.linspace(lb[1],ub[1],n2)
	x3 = np.linspace(lb[2],ub[2],n3)
	[X1,X2, X3] = np.meshgrid(x1,x2,x3)
	Xm = np.concatenate([X1.reshape((-1, 1)), X2.reshape((-1, 1)), X3.reshape((-1, 1))], axis=1)
	return Xm

