import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from jax import random, vmap, grad, jit
# from jax import random

# from torch.utils import data
import matplotlib.pyplot as plt
# import numpy as np
# import scipy.io as scio
# import h5py
# import time
import os, sys

import time
# from jax.ops import index_update, index


current_script_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
# Now add the parent directory to sys.path
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)


from analytical_sol_ade_1d import c_ext_fun
from core.shape_function_jax_1d import shape2_vmap, shape3_vmap, domain
from sub_Utilities.sub_func_plot import sub_plt_surf2D
from core.time_dependent_NIM import NIM_time_dependent
# from Model_MLPG import VPINN_MLPG

from jax import config

# config.update('jax_enable_x64', True)
    


T = 1.0

lb=np.array([-1.0, 0.0])
ub=np.array([1.0, T])

gamma = 0.1
epsilon = gamma/(np.pi)
V = 1.0


delta_test = 0.01
xtest = np.linspace(-1,1,256) 
ttest = np.arange(0, T+delta_test, delta_test)

xtest_long = np.tile(xtest.reshape(1,-1), (ttest.shape[0],1)).reshape(-1,1)
ttest_long = np.tile(ttest.reshape(-1,1), (1,xtest.shape[0])).reshape(-1,1)

XT_test = np.hstack((xtest_long, ttest_long))

c_test = c_ext_fun(XT_test, epsilon, V)


sub_plt_surf2D(lb,ub,XT_test, c_test , 'all', savefig='exact',visual=None,plt_eps=1,zlim=['None'])



#######################################################################

Nx=81
nodes=Nx

Nx_r = 81

Xi = np.linspace(lb[0], ub[0], Nx).reshape(-1,1)

Xi_rows = np.linspace(lb[0], ub[0], Nx_r).reshape(-1,1)




path_f='Results' + '/' + 'ADE_forward/'  + str(Nx) + '_' + str(Xi_rows.shape[0]) + '/'
if not os.path.exists(path_f):
    	os.makedirs(path_f)


type='cubic'
# type='gaussian'
# type='legendre'

basis='cubic'
if basis=='quadratic':
    dmax=2.5
    shape_fun=shape2_vmap
    # shape_fun=shape2_all
elif basis=='cubic':
    dmax=3.5
    shape_fun=shape3_vmap

dm = dmax*((ub[0]-lb[0])/(Nx-1))*np.ones((1,nodes))

############################################################

key = random.PRNGKey(7654321)

N_train =100 
m = 1 


key = random.PRNGKey(0) # use different key for generating test data 
keys = random.split(key, N_train)


t_seires = np.linspace(lb[1], ub[1], N_train).reshape(-1,1)


X_bc = np.vstack((lb[0], ub[0]))


c_ext0 = c_ext_fun(np.hstack((Xi_rows,np.zeros_like(Xi_rows))), epsilon, V).reshape(-1,1)



############################Save################################



# config.update('jax_enable_x64', True)
in_domain = vmap(domain, (0, None, None))(Xi_rows[:,0].reshape(-1,1),Xi,dm)
max_v = max(np.sum(in_domain, axis=1))

def calculate_shape_function(point_group):
    in_domain = vmap(domain, (0, None, None))(point_group,Xi,dm)
    index_ind = -1 * np.ones((in_domain.shape[0], max_v)).astype(np.int32)
    size_ind = np.zeros(in_domain.shape[0]).astype(np.int32)
    for i in range(in_domain.shape[0]):
        current_indices = np.where(in_domain[i, :])[0]
        l = current_indices.size 
        index_ind = index_ind.at[i, :l].set(current_indices)
        size_ind = size_ind.at[i].set(l)

    valid_mask = index_ind != -1

    phi, dphix, dphixx = vmap(shape_fun, (0, 0, 0, None, None, None)) (point_group, index_ind, size_ind, Xi, dm, max_v)
    phi = np.where(valid_mask, phi, 0)
    dphix = np.where(valid_mask, dphix, 0)
    dphixx = np.where(valid_mask, dphixx, 0)

    return phi, dphix, dphixx, index_ind

PHI_all, DPHIX_all, DPHIXX_all, index_all = calculate_shape_function(Xi_rows.reshape(-1,1))
PHI_test, DPHIX_test, _, index_test = calculate_shape_function(xtest.reshape(-1,1))
PHI_bc, _, _, index_bc = calculate_shape_function(X_bc.reshape(-1,1))
# config.update('jax_enable_x64', False)





layers = [m, 30, 30, Xi.shape[0]]


model = NIM_time_dependent(layers)


model.setup_data(PHI_all, DPHIX_all, DPHIXX_all,index_all,
                 PHI_test, index_test, PHI_bc, index_bc,
                 c_test, ttest, xtest, t_seires, N_train,
                 c_ext0, lb, V, epsilon, max_v)

batch_size = N_train


params_c = model.train(batch_size, [0,1], nIter=100000)
flat_params, _  = ravel_pytree(params_c)
np.save(path_f+'adv_params.npy', flat_params)
np.save(path_f+'adv_loss_all.npy', model.loss_log)
np.save(path_f+'adv_loss_res_c.npy', model.loss_res_c_log)
np.save(path_f+'adv_loss_ics_c.npy', model.loss_ics_c_log)
np.save(path_f+'adv_loss_bcs_c.npy', model.loss_bcs_c_log)
###################################################

# # Save the trained model

# Restore the trained model
flat_params = np.load(path_f+'adv_params.npy')
params_c = model.unravel_params(flat_params)


c_pred = model.predict_s(params_c, ttest.reshape(-1,1))

errx=abs(c_pred.reshape(-1,1)-c_test.reshape(-1,1))

print('Final Mean error: {0: .5e}'. format((errx.sum())/errx.shape[0])) 

# errors = np.linalg.norm(c_test.reshape(-1,1)-c_pred.reshape(-1,1),2)/np.linalg.norm(c_test,2)
# print(errors)

sub_plt_surf2D(lb,ub,XT_test,c_test.reshape(-1,1),'all_t',savefig=path_f+'exact',visual=None,plt_eps=0,zlim=['None'])	
sub_plt_surf2D(lb,ub,XT_test,c_pred.reshape(-1,1),'all_t',savefig=path_f+'prediction',visual=None,plt_eps=0,zlim=['None'])	
sub_plt_surf2D(lb,ub,XT_test,abs(c_test.reshape(-1,1)-c_pred.reshape(-1,1)),'all_t',savefig=path_f+'prediction_err',visual=None,plt_eps=0,zlim=['None'])	

# In[ ]:

loss_all_log = np.load(path_f+'adv_loss_all.npy')
loss_res_c_log = np.load(path_f+'adv_loss_res_c.npy')
loss_ics_c_log = np.load(path_f+'adv_loss_ics_c.npy')
loss_bcs_c_log = np.load(path_f+'adv_loss_bcs_c.npy')


# Plot for loss function
plt.figure(figsize = (6,5))
# plt.plot(model.loss_log, lw=2)
plt.plot(loss_all_log, lw=2, label='all')
plt.plot(loss_res_c_log, lw=2, label='res_c')
plt.plot(loss_ics_c_log, lw=2, label='ics_c')
plt.plot(loss_bcs_c_log, lw=2, label='bcs_c')


plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig(path_f+'loos_plot.png')
