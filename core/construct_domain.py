import jax.numpy as np
from scipy.special import jacobi
from scipy.special import roots_jacobi

def domain(gpos,x,dm):
    x=x.reshape(1,-1)
    nnodes=x.shape[1]
    dif = gpos * (np.ones((1,nnodes)))-x
    in_domain = np.all((dm - np.abs(dif)) >= 1e-8, axis=0)
    return in_domain


def cubwgt_vmap(dif,dm_v, max_v):
    
    """ Compute the cubic B-spline kernel function for shape function"""

    w,dwdx = np.zeros([1,max_v]), np.zeros([1,max_v])
    dwdxx= np.zeros([1,max_v])
    dwdxxx= np.zeros([1,max_v])

    ax = dm_v[0] 
    drdx = np.sign(dif[0])/ax 
    rx =  np.abs(dif[0])/ax 

    condition1 = rx > 1
    condition2 = rx <= 0.5
    condition3 = 0.5 < rx 

    w = np.where(condition2, (2/3) - 4*rx*rx + 4*rx **3 , w)
    dwdx = np.where(condition2, (-8*rx + 12*rx **2)*drdx, dwdx)
    dwdxx = np.where(condition2, (-8 + 24*rx)*drdx **2, dwdxx)
    dwdxxx = np.where(condition2, 24*drdx **3, dwdxxx)

    w = np.where(condition3, (4/3)-4*rx+4*rx*rx -(4/3)*rx**3, w)
    dwdx = np.where(condition3, (-4 + 8*rx - 4*rx **2)*drdx, dwdx)
    dwdxx = np.where(condition3, (8 - 8*rx)*drdx **2, dwdxx)
    dwdxxx = np.where(condition3, - 8*drdx **3, dwdxxx)

    w = np.where(condition1, 0, w)
    dwdx = np.where(condition1, 0, dwdx)
    dwdxx = np.where(condition1, 0, dwdxx)
    dwdxxx = np.where(condition1, 0, dwdxxx)

    return w,dwdx,dwdxx,dwdxxx




def weight_func(xg, xi, r): 
    """ Compute the cubic B-spline test function"""
    Weight_x, Weight_dx= np.zeros((xg.shape[0],1)), np.zeros((xg.shape[0],1))
    nv = xi
    ii=-1
    for gpos in xg:
        ii=ii+1
        dif = gpos-nv
        # w,dwdx,_= cubwgt_one(dif,r)
        w,dwdx,_,_= cubwgt_vmap(dif,r.reshape(-1), 1)
        Weight_x = Weight_x.at[ii,0].set(w.squeeze())
        Weight_dx = Weight_dx.at[ii,0].set(dwdx.squeeze())
        # Weight_x[ii,0], Weight_dx[ii,0]=w,dwdx
    return Weight_x, Weight_dx

def gauss_lobatto_jacobi_weights(Q):
    """ 
    Compute Gauss lobatto quadrature points and weights between [-1,1]

    --- Inputs: the number of quadrature points ---
    --- Outputs: the quadrature points and weights ---

    """
    X = roots_jacobi(Q-2, 1, 1)[0]
    X = np.concatenate((np.array([-1]), np.array(X), np.array([1])))
    W0 = jacobi(Q-1, 0, 0)(X)
    W = 2 / ((Q-1) * Q * W0**2)

    return X, W


def construct_subdomain_cubic(Xi_rows, lb, ub, rx, part1, quad):
    """ Compute quadrature points and weights for all subdomains """
    
    X_quad, WX_quad = gauss_lobatto_jacobi_weights(quad)


    x_quad  = X_quad.reshape(-1,1)

    w_quad  = WX_quad.reshape(-1,1)


    W_all, DWX_all = [], []
    W_ind, DWX_ind = np.zeros((quad * part1, 1)), np.zeros((quad * part1, 1))
    W_left_all = []
    W_right_all = []


    subdomain = []
    subdomain_right = []
    subdomain_left = []

    List_right = []
    List_left = []

    ii=-1

    for xi in Xi_rows:
        ii=ii+1


        left=max(xi[0]-rx,lb[0])
        right=min(xi[0]+rx,ub[0])


        lenx=(right-left)/part1

        x_quad_element, w_quad_element, jacobian_element=[], [], []
        for ll in range(part1):
            left_l=left+ll*lenx
            x_quad_element.append(left_l + lenx/2*(x_quad+1))

            jacobian_element.append(np.ones_like(x_quad)*(lenx/2))
            w_quad_element.append(w_quad)

        x_quad_element, w_quad_element, jacobian_element=np.vstack(x_quad_element), np.vstack(w_quad_element), np.vstack(jacobian_element)
        subdomain.append(np.hstack((x_quad_element,w_quad_element,jacobian_element)))

        Xf=subdomain[ii]

        W_ind, DWX_ind= weight_func(Xf, xi, rx)


        for side in range(2):
            count=0
            if side==0:
                if abs(left-lb[0])<1e-5:
                    n=-1
                    gsb=lb[0]
                    count=count+1
                    W_left_ind,_ = weight_func(gsb.reshape(-1,1), xi, rx)
                    W_left_all.append(W_left_ind)


                    subdomain_left.append(gsb)
                    List_left.append(ii)

            if side==1:
                if abs(right-ub[0])<1e-5:
                    n=1
                    gsb=ub[0]
                    count=count+1

                    W_right_ind,_= weight_func(gsb.reshape(-1,1), xi, rx)
                    W_right_all.append(W_right_ind)

                    subdomain_right.append(gsb)
                    List_right.append(ii)

        W_all.append(W_ind)
        DWX_all.append(DWX_ind)

    W_all, DWX_all, W_left_all, W_right_all = np.vstack(W_all), np.vstack(DWX_all), np.vstack(W_left_all), np.vstack(W_right_all)

    subdomain, subdomain_left, subdomain_right  = np.vstack(subdomain), np.vstack(subdomain_left), np.vstack(subdomain_right)
    List_right, List_left  = np.array(List_right), np.array(List_left)
            

    return (W_all, DWX_all, W_left_all, W_right_all,
            subdomain, subdomain_left, subdomain_right,
                List_right, List_left)
