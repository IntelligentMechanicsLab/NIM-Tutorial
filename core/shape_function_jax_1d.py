
import jax
import jax.numpy as np


def domain(gpos,x,dm):
    x=x.reshape(1,-1)
    nnodes=x.shape[1]
    dif = gpos * (np.ones((1,nnodes)))-x
    in_domain = np.all((dm - np.abs(dif)) >= 1e-8, axis=0)
    return in_domain


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


def shape2_singular_vmap(gpos, index_ind_each, size_ind_each, Xi, dm, max_v, EBC):
    """ Compute the Singular RK shape function"""
    
    x_v = Xi[index_ind_each,:]
    won = np.ones([1,max_v])
    zro=np.zeros([1,max_v])
    nv = x_v.reshape(1,max_v)
    dif = gpos*won-nv

    dm_v = dm[:,index_ind_each]

    w,dwdx,_, _= cubwgt_vmap(dif, dm_v, max_v)
    coe=0.5

    small_diff = np.abs(dif[0, :]) < 1e-10

    dist = np.where(small_diff, 1e-16, ((dif[0, :] / dm_v[0, :])**2)**coe)
    dist_dx = np.where(small_diff, 0, coe * ((dif[0, :] / dm_v[0, :])**2)**(coe - 1) * (2 * (dif[0, :] / dm_v[0, :]) / dm_v[0, :]))

    in_EBC = np.isin(index_ind_each, EBC)

    dwdx = dwdx.at[0, :].set(np.where(in_EBC, (dwdx[0, :] * dist - w[0, :] * dist_dx) / dist**2, dwdx[0, :]))
    w = w.at[0, :].set(np.where(in_EBC, w[0, :] / dist, w[0, :]))


    w = np.where(np.arange(max_v) >= size_ind_each, 0, w)
    dwdx = np.where(np.arange(max_v) >= size_ind_each, 0, dwdx)

    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:]))
    dpx = np.vstack((zro, won, 2*dif[0,:])) 


    B = p*np.vstack((w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w)) 

    aa = np.einsum('i,ji,ki->jk', w[0], p, p)
    daax = np.einsum('i,ji,ki->jk', dwdx[0], p, p) + \
        np.einsum('i,ji,ki->jk', w[0], p, dpx) + \
        np.einsum('i,ji,ki->jk', w[0], dpx, p)


    pg = np.hstack((1, 0, 0))

    aa_inv = np.linalg.inv(aa)

    r = pg @ aa_inv
    drx = -r @ daax @ aa_inv

    dbx = p * dwdx

    phi = r @ B
    dphix = drx @ B + r @ (dbx + Bdx)


    return phi, dphix


def shape2_vmap(gpos, index_ind_each, size_ind_each, Xi, dm, max_v):
    
    x_v = Xi[index_ind_each,:]
    won = np.ones([1,max_v])
    zro=np.zeros([1,max_v])
    nv = x_v.reshape(1,max_v)
    dif = gpos*won-nv

    dm_v = dm[:,index_ind_each]

    w,dwdx,dwdxx, _= cubwgt_vmap(dif, dm_v, max_v)

    w = np.where(np.arange(max_v) >= size_ind_each, 0, w)
    dwdx = np.where(np.arange(max_v) >= size_ind_each, 0, dwdx)
    dwdxx = np.where(np.arange(max_v) >= size_ind_each, 0, dwdxx)

    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:]))
    dpx = np.vstack((zro, won, 2*dif[0,:])) 
    dpxx = np.vstack((zro, zro, 2*won)) 



    B = p*np.vstack((w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w)) 
    Bdxx = dpxx*np.vstack((w, w, w)) 


    aa = np.einsum('i,ji,ki->jk', w[0], p, p)
    daax = np.einsum('i,ji,ki->jk', dwdx[0], p, p) + \
        np.einsum('i,ji,ki->jk', w[0], p, dpx) + \
        np.einsum('i,ji,ki->jk', w[0], dpx, p)

    daaxx = np.einsum('i,ji,ki->jk', dwdxx[0], p, p) + \
            np.einsum('i,ji,ki->jk', w[0], p, dpxx) + \
            np.einsum('i,ji,ki->jk', w[0], dpxx, p) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], p, dpx) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], dpx, p) + \
            2 * np.einsum('i,ji,ki->jk', w[0], dpx, dpx)



    pg = np.hstack((1, 0, 0))

    aa_inv = np.linalg.inv(aa)

    r = pg @ aa_inv
    drx = -r @ daax @ aa_inv
    drxx = -(r @ daaxx + 2 * drx @ daax) @ aa_inv

    dbx = p * dwdx
    dbxx = p * dwdxx
    dbxBdx = dpx * dwdx

    phi = r @ B
    dphix = drx @ B + r @ (dbx + Bdx)
    dphixx = drxx @ B + 2 * drx @ (dbx + Bdx) + r @ (dbxx + 2 * dbxBdx + Bdxx)


    return phi, dphix, dphixx



def shape2_all(gpos, index_ind_each, size_ind_each, Xi, dm, nodes):
    """ Compute the Singular RK shape function"""
    
    x_v = Xi[index_ind_each,:]
    won = np.ones([1,nodes])
    zro=np.zeros([1,nodes])
    nv = x_v.reshape(1,nodes)
    dif = gpos*won-nv

    dm_v = dm[:,index_ind_each]

    w,dwdx,dwdxx, _= cubwgt_vmap(dif, dm_v, nodes)

    w = np.where(np.arange(nodes) >= size_ind_each, 0, w)
    dwdx = np.where(np.arange(nodes) >= size_ind_each, 0, dwdx)
    dwdxx = np.where(np.arange(nodes) >= size_ind_each, 0, dwdxx)


    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:]))
    dpx = np.vstack((zro, won, 2*dif[0,:])) 
    dpxx = np.vstack((zro, zro, 2*won)) 



    B = p*np.vstack((w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w)) 
    Bdxx = dpxx*np.vstack((w, w, w)) 


    aa = np.einsum('i,ji,ki->jk', w[0], p, p)
    daax = np.einsum('i,ji,ki->jk', dwdx[0], p, p) + \
        np.einsum('i,ji,ki->jk', w[0], p, dpx) + \
        np.einsum('i,ji,ki->jk', w[0], dpx, p)

    daaxx = np.einsum('i,ji,ki->jk', dwdxx[0], p, p) + \
            np.einsum('i,ji,ki->jk', w[0], p, dpxx) + \
            np.einsum('i,ji,ki->jk', w[0], dpxx, p) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], p, dpx) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], dpx, p) + \
            2 * np.einsum('i,ji,ki->jk', w[0], dpx, dpx)



    pg = np.hstack((1, 0, 0))

    aa_inv = np.linalg.inv(aa)

    r = pg @ aa_inv
    drx = -r @ daax @ aa_inv
    drxx = -(r @ daaxx + 2 * drx @ daax) @ aa_inv

    dbx = p * dwdx
    dbxx = p * dwdxx
    dbxBdx = dpx * dwdx

    phi = r @ B
    dphix = drx @ B + r @ (dbx + Bdx)
    dphixx = drxx @ B + 2 * drx @ (dbx + Bdx) + r @ (dbxx + 2 * dbxBdx + Bdxx)


    return phi, dphix, dphixx




def shape3_vmap(gpos, index_ind_each, size_ind_each, Xi, dm, max_v):
    """Compute the third-order Singular RK shape function"""
    
    x_v = Xi[index_ind_each, :]
    won = np.ones([1, max_v])
    zro = np.zeros([1, max_v])
    nv = x_v.reshape(1, max_v)
    dif = gpos * won - nv

    dm_v = dm[:, index_ind_each]

    w, dwdx, dwdxx, _ = cubwgt_vmap(dif, dm_v, max_v)

    w = np.where(np.arange(max_v) >= size_ind_each, 0, w)
    dwdx = np.where(np.arange(max_v) >= size_ind_each, 0, dwdx)
    dwdxx = np.where(np.arange(max_v) >= size_ind_each, 0, dwdxx)

    # Extended polynomial basis to include cubic terms
    p = np.vstack((won, dif, dif**2, dif**3))
    dpx = np.vstack((zro, won, 2 * dif, 3 * dif**2))
    dpxx = np.vstack((zro, zro, 2 * won, 6 * dif))

    # Multiplication by weights
    B = p * np.vstack((w, w, w, w))
    Bdx = dpx * np.vstack((w, w, w, w))
    Bdxx = dpxx * np.vstack((w, w, w, w))

    # Tensor products and their derivatives
    aa = np.einsum('i,ji,ki->jk', w[0], p, p)
    daax = np.einsum('i,ji,ki->jk', dwdx[0], p, p) + \
           np.einsum('i,ji,ki->jk', w[0], p, dpx) + \
           np.einsum('i,ji,ki->jk', w[0], dpx, p)
    
    daaxx = np.einsum('i,ji,ki->jk', dwdxx[0], p, p) + \
            np.einsum('i,ji,ki->jk', w[0], p, dpxx) + \
            np.einsum('i,ji,ki->jk', w[0], dpxx, p) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], p, dpx) + \
            2 * np.einsum('i,ji,ki->jk', dwdx[0], dpx, p) + \
            2 * np.einsum('i,ji,ki->jk', w[0], dpx, dpx)

    pg = np.hstack((1, 0, 0, 0))

    aa_inv = np.linalg.inv(aa)

    r = pg @ aa_inv
    drx = -r @ daax @ aa_inv
    drxx = -(r @ daaxx + 2 * drx @ daax) @ aa_inv

    dbx = p * dwdx
    dbxx = p * dwdxx
    dbxBdx = dpx * dwdx

    phi = r @ B
    dphix = drx @ B + r @ (dbx + Bdx)
    dphixx = drxx @ B + 2 * drx @ (dbx + Bdx) + r @ (dbxx + 2 * dbxBdx + Bdxx)

    return phi, dphix, dphixx