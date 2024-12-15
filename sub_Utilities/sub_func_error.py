"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : functions for error estimation
update @ 2020.06.06
"""
import numpy as np

'''Error'''
# def sub_error_all(pred,ref):
# 	'''
# 	error_rl2: Relative L2 error
# 	error_rl2_mean: Relative L2 error with normalization substracting mean
# 	error_inf: inf norm
# 	'''
# 	error_l2 = np.linalg.norm(ref - pred, 2)/np.linalg.norm(ref, 2)
# 	mean = np.average(ref)
# 	error_l2 = np.linalg.norm(ref - pred, 2)
# 	error_rl2 = error_l2/np.linalg.norm(ref, 2)
# 	error_rl2_mean = error_l2/np.linalg.norm(ref-mean, 2)
# 	error_inf = np.linalg.norm(ref - pred, np.inf)
# 	return error_rl2, error_rl2_mean, error_inf

# Corrected @ 2020.06.18. add error_l2 as output
def sub_error_all(ref,pred):
	'''
	error_rl2: Relative L2 error
	error_rl2_mean: Relative L2 error with normalization substracting mean
	error_inf: inf norm
	'''
	mean = np.average(ref)
	error_l2 = np.linalg.norm(ref - pred, 2)
	error_rl2 = error_l2/np.linalg.norm(ref, 2)
	error_rl2_mean = error_l2/np.linalg.norm(ref-mean, 2)
	error_inf = np.linalg.norm(ref - pred, np.inf)
	return error_rl2, error_rl2_mean, error_inf, error_l2