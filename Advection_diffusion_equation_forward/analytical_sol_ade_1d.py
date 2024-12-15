from jax import config, vmap
import jax.numpy as np
def u_initial(x, t):
    utemp = - np.sin(np.pi*x)
    return utemp

def u_ext_part(x, t, epsilon, V, trunc = 800):
    """
    Function to compute the analytical solution as a Fourier series expansion.
    Inputs:
        x: column vector of locations
        t: column vector of times
        trunc: truncation number of Fourier bases
    """
    config.update('jax_enable_x64', True)
    # Series index:
    p = np.arange(0, trunc+1.0)
    p = np.reshape(p, [1, trunc+1])
    
    D = epsilon
    c0 = 16*np.pi**2*D**3*V*np.exp(V/D/2*(x-V*t/2))                           # constant
    
    c1_n = (-1)**p*2*p*np.sin(p*np.pi*x)*np.exp(-D*p**2*np.pi**2*t)           # numerator of first component
    c1_d = V**4 + 8*(V*np.pi*D)**2*(p**2+1) + 16*(np.pi*D)**4*(p**2-1)**2     # denominator of first component
    c1 = np.sinh(V/D/2)*np.sum(c1_n/c1_d, axis=-1, keepdims=True)             # first component of the solution
    
    c2_n = (-1)**p*(2*p+1)*np.cos((p+0.5)*np.pi*x)*np.exp(-D*(2*p+1)**2*np.pi**2*t/4)
    c2_d = V**4 + (V*np.pi*D)**2*(8*p**2+8*p+10) + (np.pi*D)**4*(4*p**2+4*p-3)**2
    c2 = np.cosh(V/D/2)*np.sum(c2_n/c2_d, axis=-1, keepdims=True)       # second component of the solution
    
    c = c0*(c1+c2).squeeze()
    config.update('jax_enable_x64', False)
    return c

def c_ext_fun(XT_test, epsilon, V):
    xtest_long, ttest_long = XT_test[:,0], XT_test[:,1]
    ce = vmap(u_ext_part,  (0, 0, None, None))(xtest_long, ttest_long, epsilon, V)
    ce = ce.at[ttest_long == 0].set(u_initial(xtest_long[ttest_long == 0], ttest_long[ttest_long == 0]))
    return  ce