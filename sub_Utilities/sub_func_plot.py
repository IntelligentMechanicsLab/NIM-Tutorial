"""
@author     : Qizhi He @ PNNL (qizhi.he@pnnl.gov)
Decription  : Customized Functions for ploting
update @ 2020.02.12
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from drawnow import drawnow, figure
from matplotlib import collections as mc, patches as mpatches, cm
from scipy.spatial import Delaunay
# import alphashape
import matplotlib.tri as mtri
######################################################################
############################# 2D Plotting ############################
######################################################################  

	    
'''Plot the distribution of data points'''
def sub_plt_pts2D(Xm,savefig=None,visual=None,title=None):
	def draw():
		fig = plt.figure()
		plt.plot(Xm[:,0], Xm[:,1], 'ro', markersize = 1)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)
		if title:
			plt.title(title, fontsize=16)
		# esle:
		# 	plt.title('$Collocation Points$', fontsize=16)
		
		fig.tight_layout()
		if savefig:
			path_fig_save = savefig+'map_mesh'
			fig.savefig(path_fig_save+'.png',dpi=200)
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_pts3D(Zm,savefig=None,visual=None):
	def draw():
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# ax = plt.axes(projection='3d')
		# plt.plot(Zm[:,0], Zm[:,1], Zm[:,2], 'ro', markersize = 1)
		ax.scatter3D(Zm[:,0], Zm[:,1], Zm[:,2],'ro', cmap='Greens')
		# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
		ax.set_xlabel('$x_1$', fontsize=16)
		ax.set_ylabel('$x_2$', fontsize=16)
		ax.set_zlabel('$t$', fontsize=16)
		plt.title('$Collocation Points$', fontsize=16)
		fig.tight_layout()
		if savefig:
			path_fig_save = savefig+'map_collo'
			fig.savefig(path_fig_save+'.png',dpi=200)
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_pts3D_pre(Zm,title=None,savefig=None,visual=None):
	def draw():
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# ax = plt.axes(projection='3d')
		# plt.plot(Zm[:,0], Zm[:,1], Zm[:,2], 'ro', markersize = 1)
		ax.scatter3D(Zm[:,0], Zm[:,1], Zm[:,2],'ro', cmap=cm.jet)
		ax.set_xlabel('$x_1$', fontsize=16)
		ax.set_ylabel('$x_2$', fontsize=16)
		# ax.set_zlabel('$t$', fontsize=16)
		if title is not None:
			plt.title(title, fontsize=16)
		ax.view_init(azim = 60,elev = 25) # azim: the angle to x-axis in x-y plan
		fig.tight_layout()
		if savefig:
			plt.savefig(savefig+'.png',dpi=300)
		return fig
	if visual:
		drawnow(draw)
	else:
		fig = draw()
		fig.clf()
		plt.close()

def sub_plt_pts3D_presol(Zm,Zsol,title=None,savefig=None,visual=None):
	def draw():
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		# ax = plt.axes(projection='3d')
		# plt.plot(Zm[:,0], Zm[:,1], Zm[:,2], 'ro', markersize = 1)
		ax.scatter3D(Zm[:,0], Zm[:,1], Zm[:,2],'ro', cmap=cm.jet)
		ax.scatter3D(Zsol[:,0], Zsol[:,1], Zsol[:,2],'bo', cmap=cm.viridis)
		# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');
		ax.set_xlabel('$x_1$', fontsize=16)
		ax.set_ylabel('$x_2$', fontsize=16)
		# ax.set_zlabel('$t$', fontsize=16)
		if title is not None:
			plt.title(title, fontsize=16)
		ax.view_init(azim = 60,elev = 25) # azim: the angle to x-axis in x-y plan
		fig.tight_layout()
		if savefig:
			plt.savefig(savefig+'.png',dpi=300)
		return fig
	if visual:
		drawnow(draw)
	else:
		fig = draw()
		fig.clf()
		plt.close()

'''Surf plot of Xm and U_pred'''
def sub_plt_surf2D(lb,ub,Xm1,U_pred,t,savefig=None, visual=None,plt_eps=None,zlim=None):
	def draw():
		nn = 200
		x = np.linspace(lb[0], ub[0], nn)
		y = np.linspace(lb[1], ub[1], nn)
		XX, YY = np.meshgrid(x,y)

		# Xm1=Xm1.detach().numpy()
		# U_pred=U_pred.detach().numpy()

		# U_pred1=U_pred.cpu()
		# U_pred2=U_pred1.detach().numpy()
		U_pred2=U_pred
		U_plot = griddata(Xm1, U_pred2.flatten(), (XX, YY), method='cubic')
		fig = plt.figure()

		plt.pcolor(XX, YY, U_plot, cmap='viridis')
		# plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
		if zlim:
			if zlim[0] != 'None':
				plt.clim(*zlim)
		plt.jet()
		cb = plt.colorbar()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)

		
		# cb = plt.colorbar(im, orientation="horizontal", pad=0.15)
		# cb.set_label(label='Temperature ($^{\circ}$C)', size='large', weight='bold')
		# cb.ax.tick_params(labelsize='large')
		cb.ax.tick_params(labelsize=16)

		# plt.title('$C(x_1,x_2)$', fontsize=16)
		# plt.axes().set_aspect('equal')
		fig.tight_layout()
		plt.axis('equal')
		# plt.axis('off')
		
		# plt.gca().invert_yaxis() 
		# path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
		# # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
		# fig.savefig(path_fig_save+'.png',dpi=300)
		
		if savefig:
			path_fig_save = savefig+'_t_'+str(t)
			# path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()


def sub_plt_surf2D_hole(lb,ub,Xm1,U_pred,savefig=None, visual=None,plt_eps=None,zlim=None):
	def draw():
		nn = 200
		x = np.linspace(lb[0], ub[0], nn)
		y = np.linspace(lb[1], ub[1], nn)
		XX, YY = np.meshgrid(x,y)

		# Xm1=Xm1.detach().numpy()
		# U_pred=U_pred.detach().numpy()

		# U_pred1=U_pred.cpu()
		# U_pred2=U_pred1.detach().numpy()
		U_pred2=U_pred
		U_plot = griddata(Xm1, U_pred2.flatten(), (XX, YY), method='cubic')
		fig = plt.figure()
		U_plot[XX**2+YY**2<0.1**2]=np.nan
		plt.gca().set_axis_off()
		plt.pcolor(XX, YY, U_plot, cmap='viridis')
		# plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
		if zlim:
			if zlim[0] != 'None':
				plt.clim(*zlim)
		plt.jet()
		cb = plt.colorbar()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)

		
		# cb = plt.colorbar(im, orientation="horizontal", pad=0.15)
		# cb.set_label(label='Temperature ($^{\circ}$C)', size='large', weight='bold')
		# cb.ax.tick_params(labelsize='large')
		cb.ax.tick_params(labelsize=16)

		# plt.title('$C(x_1,x_2)$', fontsize=16)
		# plt.axes().set_aspect('equal')
		fig.tight_layout()
		plt.axis('equal')
		# path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
		# # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
		# fig.savefig(path_fig_save+'.png',dpi=300)
		
		if savefig:
			path_fig_save = savefig
			# path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()



def sub_plt_surf2D_cylinder(lb, ub, Xi_post, u_pred, v_pred, savefig, num=0, s=5, scale=1):
    ''' Plot deformed plate (set scale=0 want to plot undeformed contours)
    '''
    xmin, ymin=lb[0], lb[1]
    xmax, ymax=ub[0], ub[1]
    x_pred, y_pred = Xi_post[:,[0]], Xi_post[:,[1]]
    # [x_star, y_star, u_star, v_star, s11_star, s22_star, s12_star] = preprocess(
    #     '../FEM_result/Quarter_plate_hole_dynamic/ProbeData-' + str(num) + '.mat')       # FE solution

    #
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    fig.subplots_adjust(hspace=0.2, wspace=0.3)
    
    # cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
    #                       cmap='rainbow', marker='o', s=s, vmin=0, vmax=0.04)
    cf = ax[0].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=u_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    ax[0].axis('square')
    for key, spine in ax[0].spines.items():
        if key == 'right' or key == 'top' or key == 'left' or key == 'bottom':
            spine.set_visible(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0].set_title(r'$u$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[0])
    cbar.ax.tick_params(labelsize=14)

    # cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=v_pred, alpha=0.7, edgecolors='none',
    #                       cmap='rainbow', marker='o', s=s, vmin=-0.01, vmax=0)
    cf = ax[1].scatter(x_pred + u_pred * scale, y_pred + v_pred * scale, c=v_pred, alpha=0.7, edgecolors='none',
                          cmap='rainbow', marker='o', s=s)
    for key, spine in ax[1].spines.items():
        if key in ['right', 'top', 'left', 'bottom']:
            spine.set_visible(False)
    ax[1].axis('square')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[1].set_title(r'$v$-PINN', fontsize=16)
    cbar = fig.colorbar(cf, fraction=0.046, pad=0.04, ax=ax[1])
    cbar.ax.tick_params(labelsize=14)
    #

    plt.savefig(savefig+'uv_comparison_' + str(num) + '.png', dpi=200)
    plt.close('all')


def sub_plt_surf3D(lb,ub,Xm1,U_pred,savefig=None, visual=None,plt_eps=None,zlim=None):
	def draw():
		nn = 200
		x = np.linspace(lb[0], ub[0], nn)
		y = np.linspace(lb[1], ub[1], nn)
		XX, YY = np.meshgrid(x,y)

		U_pred2=U_pred
		U_plot = griddata(Xm1, U_pred2.flatten(), (XX, YY), method='cubic')
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(XX, YY, U_plot)

		ax.set_xlabel('X', fontsize=16)
		ax.set_ylabel('Y', fontsize=16)
		ax.set_zlabel('Z', fontsize=16)
		

		fig.tight_layout()

		# def onclick(event):
		# 	if event.button == 1:
		# 	# Check if the click happened within the 3D plot
		# 		if event.inaxes == ax:
		# 			# Find the x, y, z coordinates of the clicked point
		# 			x, y, z = event.xdata, event.ydata, event.xdata
		# 			# x, y, z = ax.format_coord(x, y)

		# 			# Print the coordinates
		# 			print(f'x = {x}, y = {y}, z = {z}')

		# fig.canvas.mpl_connect('button_press_event', onclick)
		if savefig:
			path_fig_save = savefig
			# path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		# plt.show()
		# fig.clf()
		return ax
	if visual:
		drawnow(draw)
	else:
		ax=draw()
	return ax

def sub_plt_surf2D_tri(lb,ub,Xm1,U_pred,savefig=None, visual=None,plt_eps=None,zlim=None):
	def draw():

		alpha_shape1 = alphashape.alphashape(Xm1, 1.0)
		tri = Delaunay(alpha_shape1.exterior.coords)
		triangles = tri.simplices
		triang = mtri.Triangulation(Xm1[:,0], Xm1[:,1], triangles)
		# alpha_shape.show()

		fig = plt.figure()
		# ax = plt.axes(projection='3d')

		# ax.plot_trisurf(x, y, facecolors=plt.cm.jet(data), alpha=0.5)
		# ax.plot_trisurf(Xm1[:,0], Xm1[:,1], U_pred.ravel(), triangles=triangles, cmap='viridis', linewidth=0.2)

		plt.tripcolor(triang, U_pred.reshape(-1), cmap='viridis')
		# ax.tricontourf(alpha_shape1, U_pred,  cmap="Oranges")
		# plt.pcolor(XX, YY, U_plot, cmap='viridis')
		plt.tripcolor(tri.points[:,0], tri.points[:,1], tri.simplices, U_pred.ravel(), cmap='viridis')
		
		if zlim:
			if zlim[0] != 'None':
				plt.clim(*zlim)
		plt.jet()
		cb = plt.colorbar()
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel('$x_1$', fontsize=16)
		plt.ylabel('$x_2$', fontsize=16)

		
		# cb = plt.colorbar(im, orientation="horizontal", pad=0.15)
		# cb.set_label(label='Temperature ($^{\circ}$C)', size='large', weight='bold')
		# cb.ax.tick_params(labelsize='large')
		cb.ax.tick_params(labelsize=16)

		# plt.title('$C(x_1,x_2)$', fontsize=16)
		# plt.axes().set_aspect('equal')
		fig.tight_layout()
		plt.axis('equal')
		# path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
		# # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
		# fig.savefig(path_fig_save+'.png',dpi=300)
		
		if savefig:
			path_fig_save = savefig
			# path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()

def sub_plt_surf2D_wpt(lb,ub,Xm,U_pred,output=None,visual=None,plt_eps=None,points=None, cmin=None, cmax=None, title=None):
	# def draw():
	nn = 200
	x = np.linspace(lb[0], ub[0], nn)
	y = np.linspace(lb[1], ub[1], nn)
	XX, YY = np.meshgrid(x,y)

	U_plot = griddata(Xm, U_pred.flatten(), (XX, YY), method='cubic')
	# fig = plt.figure(figsize=(5, 5))
	fig = plt.figure()

	plt.pcolor(XX, YY, U_plot, cmap='viridis')
	if points is not None:
		plt.plot(points[:,0], points[:,1], 'ko', markersize = 1.0)
	
	plt.clim([cmin, cmax])
	plt.jet()
	plt.colorbar()
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.xlabel('$x_1$', fontsize=16)
	plt.ylabel('$x_2$', fontsize=16)
	if title is not None:
		plt.title(title, fontsize=14)
	# plt.axes().set_aspect('equal')
	fig.tight_layout()
	plt.axis('equal')
	
	if output is not None:
		fig.savefig(output+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(output))
	fig.clf()

'''2D Voronoi Plot'''
def plotPatch(patch, value, points=None, output=None, cmin=None, cmax=None):
	fig, ax = plt.subplots(figsize=(5, 5))
	p = mc.PatchCollection(patch, cmap=cm.jet)
	p.set_array(value)
	p.set_clim([cmin, cmax])
	ax.add_collection(p)
	if points is not None:
		ax.plot(*points, 'ko', markersize=0.5)	

	ax.axis('off')
	ax.set_aspect('equal')
	ax.autoscale(tight=True)
	fig.tight_layout()
	fig.colorbar(p)
	fig.show()
	if output is not None:
	  	fig.savefig(output, dpi=300)


######################################################################
############################# 1D Plotting ############################
######################################################################  

def sub_plt_cuvlog(loss,ylabelx=None,savefig=None, visual=None,plt_eps=None):
	def draw():
		fig = plt.figure()
		plt.semilogy(loss)
		plt.xlabel('Iteration', fontsize=16)
		plt.ylabel(ylabelx,fontsize = 16)
		fig.tight_layout()
		plt.axis('equal')
		if savefig:
			path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	if visual:
		drawnow(draw)
	else:
		draw()
def sub_plt_cuvlog_mul(lossM,labels=None,xlable=None,ylabel=None,savefig=None, plt_eps=None,plt_pdf=None,fontsize=None):
	num_line = len(labels)
	fig = plt.figure()
	# plt.semilogy(lossM[0],label=labels[0])
	# plt.semilogy(lossM[1],label=labels[1])
	# plt.semilogy(lossM[2],label=labels[2])
	if num_line==1:
		plt.semilogy(lossM,label=labels)
	elif num_line==2:
		plt.semilogy(lossM[0],label=labels[0])
		plt.semilogy(lossM[1],label=labels[1])
	elif num_line==3:
		plt.semilogy(lossM[0],label=labels[0])
		plt.semilogy(lossM[1],label=labels[1])
		plt.semilogy(lossM[2],label=labels[2])
	# plt.plot(xref[:], yref[:],'b--', linewidth=2, markersize = 5, label=yf_label)
	plt.xlabel(xlable, fontsize = fontsize)
	plt.ylabel(ylabel,fontsize = fontsize)
	legend = plt.legend(loc='best',fontsize = fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.legend(ncol=2)
	fig.tight_layout()

	if savefig:
		path_fig_save = savefig
		fig.savefig(path_fig_save+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(path_fig_save))
		if plt_pdf:
			plt.savefig('{}.pdf'.format(path_fig_save))
# def sub_plt_cuvlog_mul(lossM,labels=None,xlable=None,ylabel=None,savefig=None, plt_eps=None,plt_pdf=None,fontsize=None):
# 	num_line = len(labels)
# 	fig = plt.figure()
# 	if num_line == 4:
# 		plt.semilogy(lossM[0],label=labels[0])
# 		plt.semilogy(lossM[1],label=labels[1])
# 		plt.semilogy(lossM[2],label=labels[2])
# 		plt.semilogy(lossM[3],label=labels[3])
# 	# plt.plot(xref[:], yref[:],'b--', linewidth=2, markersize = 5, label=yf_label)
# 	plt.xlabel(xlable, fontsize = fontsize)
# 	plt.ylabel(ylabel,fontsize = fontsize)
# 	legend = plt.legend(loc='best',fontsize = fontsize)
# 	plt.xticks(fontsize=fontsize)
# 	plt.yticks(fontsize=fontsize)
	
# 	fig.tight_layout()

# 	if savefig:
# 		path_fig_save = savefig
# 		fig.savefig(path_fig_save+'.png',dpi=300)
# 		if plt_eps:
# 			plt.savefig('{}.eps'.format(path_fig_save))
# 		if plt_pdf:
# 			plt.savefig('{}.pdf'.format(path_fig_save))


def sub_plt_cuvlog2(loss,Ni,ylabelx=None,savefig=None, visual=None,plt_eps=None):
	def draw():
		fig = plt.figure()
		Nx = len(loss)
		xm_ls = np.arange(0, Ni * Nx, Ni)
		plt.semilogy(xm_ls,loss)
		plt.xlabel('Iteration', fontsize=16)
		plt.ylabel(ylabelx,fontsize = 16)
		fig.tight_layout()
		plt.axis('equal')
		if savefig:
			path_fig_save = savefig
			fig.savefig(path_fig_save+'.png',dpi=300)
			if plt_eps:
				plt.savefig('{}.eps'.format(path_fig_save))
		fig.clf()
	
	if visual:
		drawnow(draw)
	else:
		draw()

