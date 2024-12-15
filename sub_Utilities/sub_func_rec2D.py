"""
=========================================
Author: Qizhi He
Computational Mathematics Group
Advanced Computing, Mathematics and Data Division
Pacific Northwest National Laboratory 
Email: qzhe.ucsd@gmail.com; qizhi.he@pnnl.gov
=========================================
"""
# import numpy as np
import jax.numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)
# import sobol_seq  # require https://pypi.org/project/sobol_seq/

# Generate coordinates vector for uniform grids over a 2D rectangles. Order starts from left-bottom, row-wise, to right-up
def rectspace(a,b,c,d,nx,ny):
    x = np.linspace(a,b,nx)
    y = np.linspace(c,d,ny)
    [X,Y] = np.meshgrid(x,y)
    Xm = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
    return Xm, X, Y

# Normalization
def sub_rect_scale(X,Xl,Xr,scale_coe,flag):
	'''
	flag = 'scale': Map X to the parent domain  [-scale_coe,scale_coe]
	'''
	if flag == 'scale':
		len = Xr - Xl
		X_nor = 2.0 * scale_coe * (X - Xl)/len - scale_coe
	elif flag == 'shift':
		X_nor = X - 0.5
	elif flag == 'None':
		X_nor = X
	else:
		raise NotImplementedError
	return X_nor

def sub_rect_scale_Inv(X,Xl,Xr,scale_coe,flag):
	'''
	flag = 'scale': Map X to the parent domain  [-scale_coe,scale_coe]
	'''
	if flag == 'scale':
		len = Xr - Xl
		X_nor = len * (X + scale_coe) / (2.0 * scale_coe) + Xl
	elif flag == 'shift':
		X_nor = X + 0.5
	elif flag == 'None':
		X_nor = X
	else:
		raise NotImplementedError
	return X_nor
