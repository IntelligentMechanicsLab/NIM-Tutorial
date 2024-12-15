import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from jax import random, vmap, grad, jit
from jax.nn import relu
from jax.example_libraries import optimizers
from functools import partial
from tqdm import trange
import itertools
import jaxopt
import time
def MLP(layers, activation=relu):
    """Initialize and return a multi-layer perceptron (MLP)."""
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            lb_00, ub_00 = -(1 / np.sqrt(d_in)), (1 / np.sqrt(d_in)) # Xavier initialization bounds
            W = lb_00 + (ub_00 - lb_00) * jax.random.uniform(key, shape=(d_in, d_out))
            b = random.uniform(key, shape=(d_out,))
            return W, b
        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs
    return init, apply


class NIM_time_dependent_data:
    def __init__(self, layers):

        self.u_init, self.dis_apply = MLP(layers, activation=np.tanh)
        ux_params = self.u_init(rng_key=random.PRNGKey(1234))
        coe_pa = np.array([1.0])
        params = (ux_params, coe_pa)
        self.optimizer_bfgs = jaxopt.ScipyMinimize(fun=self.loss,
                                            method  = 'L-BFGS-B', maxiter = 100000,
                                            callback = self.callback,
                                            jit = True,
                                            options = {'maxfun': 100000,
                                                    'maxcor': 100,
                                                    'maxls': 100,
                                                    'ftol' : 1.0e-14,
                                                    'gtol': 1.0e-14}) 
        

        # Set optimizer
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(
            optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)
        )
        self.opt_state = self.opt_init(params)

        # For restoring trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_ics_c_log = []
        self.loss_bcs_c_log = []
        self.loss_res_c_log = []
        self.loss_data_c_log = []
        self.coe_log = []
        self.l2_log = []

    def setup_data(self, PHI_all, DPHIX_all, DPHIXX_all, index_all,
                PHI_test, index_test, PHI_bc, index_bc,
                PHI_data, index_data,t_data,c_data,
                c_test, ttest, xtest, t_seires, N_train,
                c_ext0, lb, V, epsilon, max_v):
        """
        Stores data and parameters as class attributes. 
        Indices can be used later in training steps to slice into PHI_all, DPHIX_all, DPHIXX_all.
        """
        # Storing the "full" data arrays and indexing arrays
        self.PHI_all = PHI_all
        self.DPHIX_all = DPHIX_all
        self.DPHIXX_all = DPHIXX_all
        self.index_all = index_all

        # Test data and indices
        self.PHI_test = PHI_test
        self.index_test = index_test
        
        self.PHI_data = PHI_data
        self.index_data = index_data
        self.t_data = t_data
        self.c_data = c_data
        # Boundary condition data and indices
        self.PHI_bc = PHI_bc
        self.index_bc = index_bc

        # Other parameters and data
        self.c_test = c_test
        self.ttest = ttest
        self.xtest = xtest
        self.t_seires = t_seires
        self.N_train = N_train
        self.c_ext0 = c_ext0
        self.lb = lb
        self.V = V
        self.epsilon = epsilon
        self.max_v = max_v

    def Neuro_pu(self, params, u, T, index=slice(None)):
        B = self.dis_apply(params, u).squeeze()
        B = B[index]
        outputs = (T * B).reshape(-1,self.max_v).sum(axis = 1).squeeze()
        return outputs

    def residual_net(self, params_c, coe_pa, u):
        # PDE residual: c_t + V c_x - epsilon c_xx = 0
        c_x = self.Neuro_pu(params_c, u, self.DPHIX_all, self.index_all)

        c_t = vmap(grad(self.Neuro_pu, argnums=1), in_axes=(None, None, 0, 0))(params_c, u, self.PHI_all, self.index_all)
        c_t = c_t.squeeze()

        c_xx = self.Neuro_pu(params_c, u, self.DPHIXX_all, self.index_all)

        res_c = c_t + self.V * c_x - coe_pa * c_xx
        return res_c.squeeze()

    def loss_bcs_c(self, params_c, t_select):
        # Boundary conditions loss
        c_pred0 = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, t_select, self.PHI_bc[0], self.index_bc[0])
        c_pred1 = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, t_select, self.PHI_bc[1], self.index_bc[1])
        loss_1 = np.mean(c_pred0**2) + np.mean(c_pred1**2)
        return loss_1

    def loss_ics(self, params_c, t_select):
        # Initial condition loss
        c_pred = self.Neuro_pu(params_c, t_select, self.PHI_all, self.index_all)
        loss_c = np.mean((c_pred.reshape(-1, 1) - self.c_ext0)**2)
        return loss_c

    def loss_res(self, params_c, coe_pa, t_select):
        # PDE residual loss
        pred_c = vmap(self.residual_net, (None, None, 0))(params_c,coe_pa, t_select)
        loss_c = np.mean((pred_c)**2)
        return loss_c
    def loss_data(self, params_c):

        c_pred = vmap(self.Neuro_pu, (None, 0, 0, 0))(params_c, self.t_data, self.PHI_data, self.index_data)
        # Compute loss
        loss_c = np.mean((c_pred.reshape(-1,1) - self.c_data)**2)
        return loss_c

    def loss(self, params, t_batch):
        # Total loss
        params_c, coe_pa = params
        loss_res_c = self.loss_res(params_c, coe_pa, t_batch)
        loss_ics_c = self.loss_ics(params_c, np.array([self.lb[1]]))
        loss_bcs_c = self.loss_bcs_c(params_c, self.t_seires)
        loss_data_c = self.loss_data(params_c)
        # loss = loss_res_c + 10 * loss_ics_c + loss_bcs_c
        loss =   loss_res_c + 10 * loss_ics_c + 10 * loss_bcs_c + 10 * loss_data_c
        return loss
    
    def callback(self, params):

        if self.index_opt % 100 == 0:
            elapsed_time = time.time() - self.start_time
            params_c, coe_pa = params
            loss_value = self.loss(params, self.t_seires)
            loss_ics_c_value = self.loss_ics(params_c, np.array([self.lb[1]]))
            loss_bcs_c_value = self.loss_bcs_c(params_c, self.t_seires)
            loss_data_value = self.loss_data(params_c)
            loss_res_c_value = self.loss_res(params_c, coe_pa, self.t_seires)
            c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, self.ttest, self.PHI_test, self.index_test)
            l2 = np.linalg.norm(self.c_test.reshape(-1, self.xtest.shape[0]) - c_pred, 2) / np.linalg.norm(self.c_test)
            # Store losses
            
            print('It: {0: d} | Loss: {1: .3e}| Data: {7: .3e}| Error_coe: {8: .3e}| Loss_res: {2: .3e}| Loss_is: {3: .3e}| Loss_bs: {4: .3e}|  L2: {5: .3e}| Time: {6: .2f}'.\
                format(self.index_opt, loss_value.item(), loss_res_c_value.item(), \
                       loss_ics_c_value.item(), loss_bcs_c_value.item(), \
                        l2.item(), elapsed_time, loss_data_value.item(),abs(coe_pa - self.epsilon).item())) 

            self.loss_log.append(loss_value)
            self.loss_ics_c_log.append(loss_ics_c_value)
            self.loss_bcs_c_log.append(loss_bcs_c_value)
            self.loss_res_c_log.append(loss_res_c_value)
            self.loss_data_c_log.append(loss_data_value)
            self.coe_log.append(coe_pa)
            self.l2_log.append(l2)
        self.index_opt = self.index_opt + 1

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, t_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, t_batch)
        return self.opt_update(i, g, opt_state)

    def train(self, batch_size, procedure, nIter=10000):
        # Main training loop
        if procedure[0]==1:
            key_ce = random.PRNGKey(1998)
            keys_ce = random.split(key_ce, nIter)

            pbar = trange(nIter)
            for it in pbar:
                perm = random.permutation(keys_ce[it], self.N_train)
                for bat in range(0, self.N_train, batch_size):
                    if bat + batch_size < self.N_train:
                        idx = perm[np.arange(bat, bat + batch_size)]
                    else:
                        idx = perm[np.arange(bat, self.N_train)]
                    self.opt_state = self.step(next(self.itercount), self.opt_state, self.t_seires[idx])

                if it % 100 == 0:
                    params = self.get_params(self.opt_state)
                    params_c, coe_pa = params
                    loss_value = self.loss(params, self.t_seires)
                    loss_ics_c_value = self.loss_ics(params_c, np.array([self.lb[1]]))
                    loss_bcs_c_value = self.loss_bcs_c(params_c, self.t_seires)
                    loss_data_value = self.loss_data(params_c)
                    loss_res_c_value = self.loss_res(params_c, coe_pa, self.t_seires)
                    c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, self.ttest, self.PHI_test, self.index_test)
                    l2 = np.linalg.norm(self.c_test.reshape(-1, self.xtest.shape[0]) - c_pred, 2) / np.linalg.norm(self.c_test,2)
                    # Store losses
                    self.loss_log.append(loss_value)
                    self.loss_ics_c_log.append(loss_ics_c_value)
                    self.loss_bcs_c_log.append(loss_bcs_c_value)
                    self.loss_res_c_log.append(loss_res_c_value)
                    self.loss_data_c_log.append(loss_data_value)
                    self.coe_log.append(coe_pa)
                    self.l2_log.append(l2)

                    # Print losses
                    pbar.set_postfix({'Loss': loss_value, 
                                    'loss_bcs_c' : loss_bcs_c_value, 
                                    'loss_ics_c' : loss_ics_c_value, 
                                    'loss_data_c' : loss_data_value, 
                                    'loss_physics_c': loss_res_c_value, 
                                    'loss_coe': abs(coe_pa - self.epsilon).item(), 
                                    'l2_error': l2})

            params = self.get_params(self.opt_state)

        if procedure[1]==1:
            params = self.get_params(self.opt_state)
            self.index_opt = 0
            self.start_time = time.time()
            self.solver_1_sol = self.optimizer_bfgs.run(params, self.t_seires)
            params = self.solver_1_sol.params
        return params

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, t_s):
        c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params, t_s, self.PHI_test, self.index_test)
        return c_pred

class NIM_time_dependent:
    def __init__(self, layers):

        self.u_init, self.dis_apply = MLP(layers, activation=np.tanh)
        ux_params = self.u_init(rng_key=random.PRNGKey(1234))


        self.optimizer_bfgs = jaxopt.ScipyMinimize(fun=self.loss,
                                            method  = 'L-BFGS-B', maxiter = 100000,
                                            callback = self.callback,
                                            jit = True,
                                            options = {'maxfun': 100000,
                                                    'maxcor': 100,
                                                    'maxls': 100,
                                                    'ftol' : 1.0e-14,
                                                    'gtol': 1.0e-14}) 
        

        # Set optimizer
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(
            optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)
        )
        self.opt_state = self.opt_init(ux_params)

        # For restoring trained model parameters
        _, self.unravel_params = ravel_pytree(ux_params)

        self.itercount = itertools.count()

        # Loggers
        self.loss_log = []
        self.loss_ics_c_log = []
        self.loss_bcs_c_log = []
        self.loss_res_c_log = []
        self.l2_log = []

    def setup_data(self, PHI_all, DPHIX_all, DPHIXX_all, index_all,
                PHI_test, index_test, PHI_bc, index_bc,
                c_test, ttest, xtest, t_seires, N_train,
                c_ext0, lb, V, epsilon, max_v):
        """
        Stores data and parameters as class attributes. 
        Indices can be used later in training steps to slice into PHI_all, DPHIX_all, DPHIXX_all.
        """
        # Storing the "full" data arrays and indexing arrays
        self.PHI_all = PHI_all
        self.DPHIX_all = DPHIX_all
        self.DPHIXX_all = DPHIXX_all
        self.index_all = index_all

        # Test data and indices
        self.PHI_test = PHI_test
        self.index_test = index_test

        # Boundary condition data and indices
        self.PHI_bc = PHI_bc
        self.index_bc = index_bc

        # Other parameters and data
        self.c_test = c_test
        self.ttest = ttest
        self.xtest = xtest
        self.t_seires = t_seires
        self.N_train = N_train
        self.c_ext0 = c_ext0
        self.lb = lb
        self.V = V
        self.epsilon = epsilon
        self.max_v = max_v

    def Neuro_pu(self, params, u, T, index=slice(None)):
        B = self.dis_apply(params, u).squeeze()
        B = B[index]
        outputs = (T * B).reshape(-1,self.max_v).sum(axis = 1).squeeze()
        return outputs

    def residual_net(self, params_c, u):
        # PDE residual: c_t + V c_x - epsilon c_xx = 0
        c_x = self.Neuro_pu(params_c, u, self.DPHIX_all, self.index_all)

        c_t = vmap(grad(self.Neuro_pu, argnums=1), in_axes=(None, None, 0, 0))(params_c, u, self.PHI_all, self.index_all)
        c_t = c_t.squeeze()

        c_xx = self.Neuro_pu(params_c, u, self.DPHIXX_all, self.index_all)

        res_c = c_t + self.V * c_x - self.epsilon * c_xx
        return res_c.squeeze()

    def loss_bcs_c(self, params_c, t_select):
        # Boundary conditions loss
        c_pred0 = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, t_select, self.PHI_bc[0], self.index_bc[0])
        c_pred1 = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, t_select, self.PHI_bc[1], self.index_bc[1])
        loss_1 = np.mean(c_pred0**2) + np.mean(c_pred1**2)
        return loss_1

    def loss_ics(self, params_c, t_select):
        # Initial condition loss
        c_pred = self.Neuro_pu(params_c, t_select, self.PHI_all, self.index_all)
        loss_c = np.mean((c_pred.reshape(-1, 1) - self.c_ext0)**2)
        return loss_c

    def loss_res(self, params_c, t_select):
        # PDE residual loss
        pred_c = vmap(self.residual_net, (None, 0))(params_c, t_select)
        loss_c = np.mean((pred_c)**2)
        return loss_c

    def loss(self, params_c, t_batch):
        # Total loss
        loss_res_c = self.loss_res(params_c, t_batch)
        loss_ics_c = self.loss_ics(params_c, np.array([self.lb[1]]))
        loss_bcs_c = self.loss_bcs_c(params_c, self.t_seires)
        loss = loss_res_c + 10 * loss_ics_c + loss_bcs_c
        # loss = loss_ics_c
        return loss
    
    def callback(self, params):

        if self.index_opt % 100 == 0:
            elapsed_time = time.time() - self.start_time
            loss_value = self.loss(params, self.t_seires)
            loss_ics_c_value = self.loss_ics(params, np.array([self.lb[1]]))
            loss_bcs_c_value = self.loss_bcs_c(params, self.t_seires)
            loss_res_c_value = self.loss_res(params, self.t_seires)
            c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params, self.ttest, self.PHI_test, self.index_test)
            l2 = np.linalg.norm(self.c_test.reshape(-1, self.xtest.shape[0]) - c_pred, 2) / np.linalg.norm(self.c_test)
            # Store losses
            
            print('It: {0: d} | Loss: {1: .3e}| Loss_res: {2: .3e}| Loss_is: {3: .3e}| Loss_bs: {4: .3e}|  L2: {5: .3e}| Time: {6: .2f}'.\
                format(self.index_opt, loss_value.item(), loss_res_c_value.item(), loss_ics_c_value.item(), loss_bcs_c_value.item(), l2.item(), elapsed_time)) 

            self.loss_log.append(loss_value)
            self.loss_ics_c_log.append(loss_ics_c_value)
            self.loss_bcs_c_log.append(loss_bcs_c_value)
            self.loss_res_c_log.append(loss_res_c_value)
            self.l2_log.append(l2)
        self.index_opt = self.index_opt + 1

    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, t_batch):
        params = self.get_params(opt_state)
        g = grad(self.loss)(params, t_batch)
        return self.opt_update(i, g, opt_state)

    def train(self, batch_size, procedure, nIter=10000):
        # Main training loop
        if procedure[0]==1:
            key_ce = random.PRNGKey(1998)
            keys_ce = random.split(key_ce, nIter)

            pbar = trange(nIter)
            for it in pbar:
                perm = random.permutation(keys_ce[it], self.N_train)
                for bat in range(0, self.N_train, batch_size):
                    if bat + batch_size < self.N_train:
                        idx = perm[np.arange(bat, bat + batch_size)]
                    else:
                        idx = perm[np.arange(bat, self.N_train)]
                    self.opt_state = self.step(next(self.itercount), self.opt_state, self.t_seires[idx])

                if it % 100 == 0:
                    params = self.get_params(self.opt_state)
                    params_c = params
                    # Compute losses
                    loss_value = self.loss(params, self.t_seires)
                    loss_ics_c_value = self.loss_ics(params_c, np.array([self.lb[1]]))
                    loss_bcs_c_value = self.loss_bcs_c(params_c, self.t_seires)
                    loss_res_c_value = self.loss_res(params_c, self.t_seires)
                    c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params_c, self.ttest, self.PHI_test, self.index_test)
                    l2 = np.linalg.norm(self.c_test.reshape(-1, self.xtest.shape[0]) - c_pred, 2) / np.linalg.norm(self.c_test,2)
                    # Store losses
                    self.loss_log.append(loss_value)
                    self.loss_ics_c_log.append(loss_ics_c_value)
                    self.loss_bcs_c_log.append(loss_bcs_c_value)
                    self.loss_res_c_log.append(loss_res_c_value)
                    self.l2_log.append(l2)

                    # Print losses
                    pbar.set_postfix({
                        'Loss': loss_value,
                        'loss_bcs_c': loss_bcs_c_value,
                        'loss_ics_c': loss_ics_c_value,
                        'loss_physics_c': loss_res_c_value,
                        'l2_error': l2
                    })

            params = self.get_params(self.opt_state)

        if procedure[1]==1:
            params = self.get_params(self.opt_state)
            self.index_opt = 0
            self.start_time = time.time()
            self.solver_1_sol = self.optimizer_bfgs.run(params, self.t_seires)
            params = self.solver_1_sol.params
        return params

    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, t_s):
        c_pred = vmap(self.Neuro_pu, (None, 0, None, None))(params, t_s, self.PHI_test, self.index_test)
        return c_pred
