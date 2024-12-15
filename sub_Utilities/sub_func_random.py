import numpy as np
from pyDOE import lhs   # The experimental design package for python; Latin Hypercube Sampling (LHS)
# import sobol_seq  # require https://pypi.org/project/sobol_seq/


def rectrand(lb,ub,dim,Nf,Nf_max,flag_rand=None):
    znor = lhs(dim,Nf_max)[:Nf,:]
    scale = (ub[:,np.newaxis]-lb[:,np.newaxis])
    zm = np.zeros((Nf,dim))
    for i in range(dim):
        zm[:,i] = lb[i] + (ub[i]-lb[i]) * znor[:,i]

    # znor = lhs(pro_dim,Nf_max)[:Nf,:]
    # Zm_f_train = np.zeros((Nf,pro_dim))
    # Zm_f_train[:,0:1] = lb[0]  + (ub[0]-lb[0])*znor[:,0:1]
    # Zm_f_train[:,1:2] = lb[1]  + (ub[1]-lb[1])*znor[:,1:2]
    # Zm_f_train[:,2:3] = t_min + (t_max-t_min)*znor[:,2:3]
    return znor, zm


def rectrand_reg(lb,ub,dim,Nf,Nf_max,flag_rand=None):
    znor = lhs(dim,Nf_max)[:Nf,:]
    # scale = (ub[:]-lb[:])
    zm = np.zeros((Nf,dim))
    # zm = znor
    for i in range(dim):
        # scale_i = scale[i]
        for j in range(Nf):
            zm[j,i] = lb[i] + (ub[i]-lb[i])/(Nf-1)*j
    return znor, zm

# def rectspace(a,b,c,d,nx,ny):
#     x = np.linspace(a,b,nx)
#     y = np.linspace(c,d,ny)
#     [X,Y] = np.meshgrid(x,y)
#     Xm = np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
#     return Xm

# def rectspace_dis(lb,ub,N,len_ratio=None,adjust=None,rand_rate=None):
    
#     if len_ratio:
#         ny = np.sqrt(N/len_ratio).astype(int)
#         nx = (N/ny).astype(int)
#         N_new = nx * ny
#     else:
#         ny = np.sqrt(N/2).astype(int)
#         nx = (N/ny).astype(int)
#         N_new = nx * ny        

#     a, b, c, d = lb[0], ub[0],lb[1],ub[1]
    
#     if adjust:
#         a = a + adjust
#         b = b - adjust
#         c = c + adjust
#         d = d - adjust

#     Xm = rectspace(a,b,c,d,nx,ny)
#     if rand_rate:
#         Xm[:,0:1] = Xm[:,0:1] + np.random.normal(0,rand_rate,(N_new,1))
#         Xm[:,1:2] = Xm[:,1:2] + np.random.normal(0,rand_rate,(N_new,1))        
#     return Xm, N_new

# # def rect_sob_Dis2d(Nx,Ny,N_k,ny0=None,rand_rate=None):
# #     # Quasi-random: i4_sobol_generate
# #     if ny0:
# #         ny = ny0
# #     else:
# #     	ny = np.sqrt(N_k/2).astype(int)
# #     nx = (N_k/ny).astype(int)
    
# #     if nx*ny != N_k:
# #         print('Error: nx*ny not equal to N_k, reset N_k')
# #         N_k = nx*ny

# #     sob_k = sobol_seq.i4_sobol_generate(2,Nx*Ny) # Generate a 2-Dim Vecotr 
# #     # array = sobol_seq.i4_sobol_generate(2, 10)
# #     XX, YY = np.meshgrid(sob_k[0:nx,0],sob_k[0:ny,1])
# #     Xm_k = np.zeros((N_k,2))
# #     if rand_rate:
# #         Xm_k[:,0:1] = XX.reshape((nx*ny,-1)) + np.random.normal(0,rand_rate,(nx*ny,1))
# #         Xm_k[:,1:2] = YY.reshape((nx*ny,-1)) + np.random.normal(0,rand_rate,(nx*ny,1))
# #     else:
# #         Xm_k[:,0:1] = XX.reshape((nx*ny,-1))
# #         Xm_k[:,1:2] = YY.reshape((nx*ny,-1))
# #     return Xm_k, N_k

# # def rect_sob_DisIndex(Nx,Ny,N_k,ny0=None):
# #     if ny0:
# #         ny = ny0
# #     else:
# #         ny = np.sqrt(N_k/2).astype(int)

# #     nx = (N_k/ny).astype(int)
    
# #     sob_k = sobol_seq.i4_sobol_generate(2,Nx*Ny)
# #     ldx_k = (sob_k[0:nx,0] * Nx).astype(int)
# #     ldy_k = (sob_k[0:ny,1] * Ny).astype(int)
# #     idx_k = []
# #     for i_loop in ldy_k:
# #         for j_loop in ldx_k:
# #             idx_i = i_loop * Nx + j_loop
# #             idx_k.append(idx_i)
# #     return idx_k

# def rect_PartitionedDisUni2d(N,Ns,Np_x,Np_y,rand_rate=None):
#     # for [0,1] * [0,1], partioned into Np_x * Np_y domains, and random select locations from these subdomains
#     num_par = Np_x * Np_y
#     Ns_k   = Ns // num_par
#     Ns_res = Ns % num_par

#     Index_Ns = (Ns_k * np.ones(num_par)).astype(int)
    
#     # idx = np.random.choice(num_par,Ns_res,replace=False)
#     Index_Ns[0:Ns_res] += 1 

#     xm = np.linspace(0, 1, Np_x+1)
#     ym = np.linspace(0, 1, Np_y+1)

#     XM = np.zeros((Ns,2))
#     index_cout = 0
#     for j_loop in range(0,Np_y):
#         for i_loop in range(0,Np_x):
#             k_par = j_loop * Np_x + i_loop
#             lb = np.array([xm[i_loop],ym[j_loop]])
#             ub = np.array([xm[i_loop+1],ym[j_loop+1]])
#             num_k = Index_Ns[k_par]
#             Xs_k  = lb + (ub-lb)*(np.random.uniform(0.,1.,(N,2))[:num_k,:])
#             # Xs_k  = lb + (ub-lb)*(lhs(2, N)[:num_k,:])
#             XM[index_cout:index_cout+num_k,:] = Xs_k
#             index_cout += num_k
#             # X_hbN = np.concatenate([xb1, xb3], axis = 0)   # approximating value
#     return XM    

# def find_nearestl2(XM, x):
#     # array = np.asarray(array)
#     XM0 = XM - x
#     norm_XM0 = np.linalg.norm(XM0,axis=1)
#     idx = norm_XM0.argmin()
#     return XM[idx,:], idx


# def find_nearestVec(XM, xm):
# #	(num_y,num_x) = xm.shape
#     idx = []
#     (num_y,num_x) = xm.shape
#     xm_new = np.zeros((num_y,num_x))
#     for i in range(0,num_y):
#         X_i, idx_i = find_nearestl2(XM, xm[i,:])
#         xm_new[i,:] = X_i
#         idx.append(idx_i)
#     return idx, xm_new 


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx], idx

# '''Error'''
# def error_relative(pred,ref):
#     error_l2 = np.linalg.norm(ref - pred, 2)/np.linalg.norm(ref, 2)

#     mean = np.average(ref)
#     error_l2 = np.linalg.norm(ref - pred, 2)
#     error_rl2 = error_l2/np.linalg.norm(ref, 2)
#     error_rl2_mean = error_l2/np.linalg.norm(ref-mean, 2)
#     error_inf = np.linalg.norm(ref - pred, np.inf)
#     return error_rl2, error_rl2_mean, error_inf

# def relative_error(pred, exact):
#     if type(pred) is np.ndarray:        
#         return np.sqrt(np.mean(np.square(pred - exact))/np.mean(np.square(exact - np.mean(exact))))
#     return tf.sqrt(tf.reduce_mean(tf.square(pred - exact))/tf.reduce_mean(tf.square(exact - tf.reduce_mean(exact))))

# def mean_squared_error(pred, exact):
#     if type(pred) is np.ndarray:
#         return np.mean(np.square(pred - exact))
#     return tf.reduce_mean(tf.square(pred - exact))