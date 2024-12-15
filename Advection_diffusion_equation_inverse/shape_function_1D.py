import numpy as np

from sub_Utilities.GaussJacobiQuadRule_V3 import Jacobi

def domain(gpos,x,dm):
    x=x.reshape(1,-1)
    nnodes=x.shape[1]
    # dif = gpos.dot(np.ones((1,nnodes)))-x
    dif = gpos * (np.ones((1,nnodes)))-x
    a = dm-np.abs(dif)
    c = a>=1e-8
    v=[]
    for i in range(nnodes):
        if c[:,i].all():
            v.append(i)
    v=np.array(v).reshape(1,len(v))
    return v
def cubwgt(dif,v,dm):
    if isinstance(v,int):
        L=1
    else:
        v=v.astype(int)
        L = np.size(v,1)
    w,dwdx = np.zeros([1,L]), np.zeros([1,L])
    dwdxx= np.zeros([1,L])
    dwdxxx= np.zeros([1,L])
    for i in range(L):
        ax = dm[0,v[0,i]] 
        drdx = np.sign(dif[0,i])/ax 
        rx =  np.abs(dif[0,i])/ax 

        #  3-spline
        if rx>1:
            wx = 0
            dwx = 0
            dwxx = 0
            dwxxx = 0
        elif rx<=0.5:
            wx = (2/3) - 4*rx*rx + 4*rx **3       
            dwx = (-8*rx + 12*rx **2)*drdx
            dwxx = (-8 + 24*rx)*drdx **2
            dwxxx = 24*drdx **3
        else:
            wx = (4/3)-4*rx+4*rx*rx -(4/3)*rx**3
            dwx = (-4 + 8*rx - 4*rx **2)*drdx
            dwxx = (8 - 8*rx)*drdx **2
            dwxxx = - 8*drdx **3

        w[0,i] = wx
        dwdx[0,i] = dwx 
        dwdxx[0,i] = dwxx 
        dwdxxx[0,i] = dwxxx

    return w,dwdx,dwdxx,dwdxxx



def weight_one(xg,xi,type,r,N_weights):
    Weight_x, Weight_dx, Weight_dy= np.zeros((xg.shape[0],N_weights**2)), np.zeros((xg.shape[0],N_weights**2)), np.zeros((xg.shape[0],N_weights**2))
    nv = xi.reshape(2,1)
    if type=='combine':
        ind=-1
        for ii in ['quadratic', 'cubic', 'quartic', 'quintic'][:N_weights]:
            for jj in ['quadratic', 'cubic', 'quartic', 'quintic'][:N_weights]:
        # for ii in ['cubic', 'quartic', 'quintic'][:N_weights]:
        #     for jj in ['cubic', 'quartic', 'quintic'][:N_weights]:
                ind=ind+1
                for kk in range(xg.shape[0]):
                    gpos=xg[kk,:]
                    dif = gpos.reshape(2,1)-nv
                    wx,dwx= spline_one(dif[0],ii,r)
                    wy,dwy= spline_one(dif[1],jj,r)
                    Weight_x[kk,ind], Weight_dx[kk,ind], Weight_dy[kk,ind]=wx*wy, dwx*wy, wx*dwy
    return Weight_x, Weight_dx, Weight_dy

def spline_one(dif, type, r):
    
    L=1
    ax = r
    drdx = np.sign(dif[0])/ax 
    rx =  np.abs(dif[0])/ax 
    if type=='quintic':
    # 5-spline
        if rx>1:
            wx = 0 
            dwx = 0 
        elif rx<=1/3:
            wx = (3-3*rx)**5 - 6*(2-3*rx)**5 + 15*(1-3*rx)**5 
            dwx = (-3*(3-3*rx)**4 + 18*(2-3*rx)**4 - 45*(1-3*rx)**4)*drdx 
        elif rx<=2/3 and rx>1/3:
            wx = (3-3*rx)**5 - 6*(2-3*rx)**5 
            dwx = (-3*(3-3*rx)**4 + 18*(2-3*rx)**4)*drdx 
        else:
            wx = (3-3*rx)**5 
            dwx = -3*(3-3*rx)**4*drdx 
        wx = wx/120
        dwx = dwx/24 

    if type=='cubic':
        #  3-spline
        if rx>1:
            wx = 0
            dwx = 0
        elif rx<=0.5:
            wx = (2/3) - 4*rx*rx + 4*rx **3       
            dwx = (-8*rx + 12*rx **2)*drdx
        else:
            wx = (4/3)-4*rx+4*rx*rx -(4/3)*rx**3
            dwx = (-4 + 8*rx - 4*rx **2)*drdx
    if type=='quartic':
        if rx>1:
            wx = 0
            dwx = 0
        else:
            wx = 1-6*rx**2+8*rx**3-3*rx**4
            dwx = (- 12*rx**3 + 24*rx**2 - 12*rx)*drdx
    if type=='quadratic':
        if rx>1:
            wx = 0
            dwx = 0
        elif rx<=1/3:
            wx = 3/4-9/4*rx**2
            dwx = -9/2*rx
        else:
            wx = 9/8*(1-rx)**2
            dwx = -9/4*(1-rx)
    return wx, dwx


def cubwgt_one(dif, type, r, beta):

    L=1
    w,dwdx= np.zeros([1,L]), np.zeros([1,L])
    dwdxx= np.zeros([1,L])
    
    ax = r

    drdx = np.sign(dif[0])/ax 

    rx =  np.abs(dif[0])/ax 

    if type=='quintic':
    # 5-spline
        if rx>1:
            wx = 0 
            dwx = 0 
            dwxx = 0 
            dwxxx = 0 
        elif rx<=1/3:
            wx = (3-3*rx)**5 - 6*(2-3*rx)**5 + 15*(1-3*rx)**5 
            dwx = (-3*(3-3*rx)**4 + 18*(2-3*rx)**4 - 45*(1-3*rx)**4)*drdx 
            dwxx = (9*(3-3*rx)**3 - 54*(2-3*rx)**3 + 135*(1-3*rx)**3)*drdx**2 
            dwxxx = (-27*(3-3*rx)**2 + 162*(2-3*rx)**2 - 405*(1-3*rx)**2)*drdx**3 
        elif rx<=2/3 and rx>1/3:
            wx = (3-3*rx)**5 - 6*(2-3*rx)**5 
            dwx = (-3*(3-3*rx)**4 + 18*(2-3*rx)**4)*drdx 
            dwxx = (9*(3-3*rx)**3 - 54*(2-3*rx)**3)*drdx**2 
            dwxxx = (-27*(3-3*rx)**2 + 162*(2-3*rx)**2)*drdx**3 
        else:
            wx = (3-3*rx)**5 
            dwx = -3*(3-3*rx)**4*drdx 
            dwxx = 9*(3-3*rx)**3*drdx**2 
            dwxxx = -27*(3-3*rx)**2*drdx**3 

        wx  = wx/120
        dwx  = dwx/24
        dwxx  = dwxx/6
        dwxxx = dwxxx/2

    if type=='cubic':
        #  3-spline
        if rx>1:
            wx = 0
            dwx = 0
            dwxx = 0
            dwxxx = 0
        elif rx<=0.5:
            wx = (2/3) - 4*rx*rx + 4*rx **3       
            dwx = (-8*rx + 12*rx **2)*drdx
            dwxx = (-8 + 24*rx)*drdx **2
            dwxxx = 24*drdx **3
        else:
            wx = (4/3)-4*rx+4*rx*rx -(4/3)*rx**3
            dwx = (-4 + 8*rx - 4*rx **2)*drdx
            dwxx = (8 - 8*rx)*drdx **2
            dwxxx = - 8*drdx **3

    if type=='quartic':
        if rx>1:
            wx = 0
            dwx = 0
            dwxx = 0
        else:
            wx = 1-6*rx**2+8*rx**3-3*rx**4
            dwx = (- 12*rx**3 + 24*rx**2 - 12*rx)*drdx
            dwxx = (- 36*rx**2 + 48*rx - 12)*drdx**2

            
    if type=='quadratic':
        if rx>1:
            wx = 0
            dwx = 0
            dwxx = 0
        elif rx<=1/3:
            wx = 3/4-9/4*rx**2
            dwx = -9/2*rx
            dwxx = -9/2
        else:
            wx = 9/8*(1-rx)**2
            dwx = -9/4*(1-rx)
            dwxx = 9/4

    w = wx
    dwdx = dwx 
    dwdxx = dwxx 

    # dwdyyy = wx*dwyyy 
    return w,dwdx,dwdxx

def weight_func(xg, xi, type, r, beta): 
    Weight_x, Weight_dx= np.zeros((xg.shape[0],1)), np.zeros((xg.shape[0],1))
    nv = xi
    ii=-1
    for gpos in xg:
        ii=ii+1
        dif = gpos-nv
        w,dwdx,_= cubwgt_one(dif,type,r,beta)
        Weight_x[ii,0], Weight_dx[ii,0]=w,dwdx
    return Weight_x, Weight_dx



def shape2(gpos,x,v,dm):
    v=v.astype(int)
    L = np.size(v,1)
    won = np.ones([1,L])
    zro=np.zeros([1,L])
    nv = x[0,v].reshape(1,L)
    dif = gpos*won-nv
    w,dwdx,dwdxx, _= cubwgt(dif,v,dm)

    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:]))


    dpx = np.vstack((zro, won, 2*dif[0,:])) 
    dpxx = np.vstack((zro, zro, 2*won)) 



    B = p*np.vstack((w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w)) 
    Bdxx = dpxx*np.vstack((w, w, w)) 



    aa = np.zeros([3,3]) 
    daax = np.zeros([3,3]) 
    daaxx = np.zeros([3,3])  

    for i in range(L):
        pi=p[:,i].reshape(3,1)
        pit=p[:,i].reshape(1,3)
        dpxi=dpx[:,i].reshape(3,1)
        dpxit=dpx[:,i].reshape(1,3)

        dpxxi=dpxx[:,i].reshape(3,1)

        dpxxit=dpxx[:,i].reshape(1,3)


        pp = pi.dot(pit)
        aa = aa+w[0,i]*pp 
        daax = daax+dwdx[0,i]*pp+w[0,i]*pi.dot(dpxit)+w[0,i]*dpxi.dot(pit)

        daaxx = daaxx + dwdxx[0,i]*pp + w[0,i]*dpxxi.dot(pit) + w[0,i]*pi.dot(dpxxit) \
                    + 2*dwdx[0,i]*dpxi.dot(pit) + 2*dwdx[0,i]*pi.dot(dpxit) + 2*w[0,i]*dpxi.dot(dpxit) 


    pg = np.hstack((1, 0, 0)).reshape(1,3)

    r = pg.dot(np.linalg.inv(aa)) 
    drx = - r.dot(daax).dot(np.linalg.inv(aa)) 
    drxx = - (r.dot(daaxx) + 2*drx.dot(daax)).dot(np.linalg.inv(aa)) 


    dbx = p*np.vstack((dwdx, dwdx, dwdx))
    dbxx = p*np.vstack((dwdxx, dwdxx, dwdxx))



    dbxBdx = dpx*np.vstack((dwdx, dwdx, dwdx))


    phi = r.dot(B) 
    dphix = drx.dot(B) + r.dot(dbx+Bdx) 

    dphixx = drxx.dot(B) + 2*drx.dot(dbx+Bdx) + r.dot(dbxx+2*dbxBdx+Bdxx) 


    # dphixx=0
    return phi, dphix, dphixx

def shape2_singular(gpos,x,v,dm,EBC):
    v=v.astype(int)
    L = np.size(v,1)
    won = np.ones([1,L])
    zro=np.zeros([1,L])
    nv = x[0,v].reshape(1,L)
    dif = gpos*won-nv
    w,dwdx,dwdxx, _= cubwgt(dif,v,dm)
    coe=0.5
    for ii in range(L):
        if v[0,ii] in EBC:
        # if v[0,ii] in [1558,1559,1560]:
            # if dif[0,ii]==0 or dif[1,ii]==0:
            if abs(dif[0,ii])<1e-10 and abs(dif[1,ii])<1e-10:
                dist=1e-16
                dist_dx, dist_dy = 0, 0
            else:
                dist=((dif[0,ii]/dm[0,ii])**2)**coe

                dist_dx=coe*((dif[0,ii]/dm[0,ii])**2)**(coe-1)*(2*(dif[0,ii]/dm[0,ii])/dm[0,ii])

            dwdx[0,ii]=(dwdx[0,ii]*dist-w[0,ii]*dist_dx)/ dist**2
            w[0,ii]=w[0,ii]/ dist

    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:]))


    dpx = np.vstack((zro, won, 2*dif[0,:])) 
    dpxx = np.vstack((zro, zro, 2*won)) 



    B = p*np.vstack((w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w)) 
    Bdxx = dpxx*np.vstack((w, w, w)) 



    aa = np.zeros([3,3]) 
    daax = np.zeros([3,3]) 
    daaxx = np.zeros([3,3])  

    for i in range(L):
        pi=p[:,i].reshape(3,1)
        pit=p[:,i].reshape(1,3)
        dpxi=dpx[:,i].reshape(3,1)
        dpxit=dpx[:,i].reshape(1,3)

        dpxxi=dpxx[:,i].reshape(3,1)

        dpxxit=dpxx[:,i].reshape(1,3)


        pp = pi.dot(pit)
        aa = aa+w[0,i]*pp 
        daax = daax+dwdx[0,i]*pp+w[0,i]*pi.dot(dpxit)+w[0,i]*dpxi.dot(pit)

        daaxx = daaxx + dwdxx[0,i]*pp + w[0,i]*dpxxi.dot(pit) + w[0,i]*pi.dot(dpxxit) \
                    + 2*dwdx[0,i]*dpxi.dot(pit) + 2*dwdx[0,i]*pi.dot(dpxit) + 2*w[0,i]*dpxi.dot(dpxit) 


    pg = np.hstack((1, 0, 0)).reshape(1,3)

    r = pg.dot(np.linalg.inv(aa)) 
    drx = - r.dot(daax).dot(np.linalg.inv(aa)) 
    drxx = - (r.dot(daaxx) + 2*drx.dot(daax)).dot(np.linalg.inv(aa)) 


    dbx = p*np.vstack((dwdx, dwdx, dwdx))
    dbxx = p*np.vstack((dwdxx, dwdxx, dwdxx))



    dbxBdx = dpx*np.vstack((dwdx, dwdx, dwdx))


    phi = r.dot(B) 
    dphix = drx.dot(B) + r.dot(dbx+Bdx) 

    dphixx = drxx.dot(B) + 2*drx.dot(dbx+Bdx) + r.dot(dbxx+2*dbxBdx+Bdxx) 


    # dphixx=0
    return phi, dphix, dphixx


def shape3(gpos,x,v,dm):
    v=v.astype(int)
    L = np.size(v,1)
    won = np.ones([1,L])
    zro=np.zeros([1,L])
    nv = x[0,v].reshape(1,L)
    dif = gpos*won-nv
    w,dwdx,dwdxx, _= cubwgt(dif,v,dm)

    p = np.vstack((won,dif,dif[[0],:]*dif[[0],:], dif[[0],:]**3))


    dpx = np.vstack((zro, won, 2*dif[0,:], 3*dif[[0],:]**2)) 
    dpxx = np.vstack((zro, zro, 2*won, 6*dif[[0],:]))  

    # WEIGHTS--W and dw are vectors

    B = p*np.vstack((w, w, w, w)) 
    Bdx = dpx*np.vstack((w, w, w, w)) 
    Bdxx = dpxx*np.vstack((w, w, w, w)) 


    aa = np.zeros([4,4]) 
    daax = np.zeros([4,4]) 
    daaxx = np.zeros([4,4])


    for i in range(L):
        pi=p[:,i].reshape(4,1)
        pit=p[:,i].reshape(1,4)
        dpxi=dpx[:,i].reshape(4,1)
        dpxit=dpx[:,i].reshape(1,4)
        dpxxi=dpxx[:,i].reshape(4,1)


        dpxxit=dpxx[:,i].reshape(1,4)
  

        pp = pi.dot(pit)
        aa = aa+w[0,i]*pp 

        daax = daax+dwdx[0,i]*pp+w[0,i]*pi.dot(dpxit)+w[0,i]*dpxi.dot(pit)

        daaxx = daaxx + dwdxx[0,i]*pp + w[0,i]*dpxxi.dot(pit) + w[0,i]*pi.dot(dpxxit) \
                    + 2*dwdx[0,i]*dpxi.dot(pit) + 2*dwdx[0,i]*pi.dot(dpxit) + 2*w[0,i]*dpxi.dot(dpxit) 


    pg = np.hstack((1, 0, 0, 0)).reshape(1,4)

    r = pg.dot(np.linalg.inv(aa)) 
    drx = - r.dot(daax).dot(np.linalg.inv(aa)) 
    drxx = - (r.dot(daaxx) + 2*drx.dot(daax)).dot(np.linalg.inv(aa)) 


    dbx = p*np.vstack((dwdx, dwdx, dwdx, dwdx))
    dbxx = p*np.vstack((dwdxx, dwdxx, dwdxx, dwdxx))



    dbxBdx = dpx*np.vstack((dwdx, dwdx, dwdx, dwdx))


    phi = r.dot(B) 
    dphix = drx.dot(B) + r.dot(dbx+Bdx) 

    dphixx = drxx.dot(B) + 2*drx.dot(dbx+Bdx) + r.dot(dbxx+2*dbxBdx+Bdxx) 

    return phi, dphix, dphixx