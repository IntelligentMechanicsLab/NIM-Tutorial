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

######################################################################
############################# 1D Plotting ############################
######################################################################  

'''Surf plot of Xm and Solution (or prediction)'''
def plot_1Dcurve(xh,yh,yh_label=None,xlable=None,ylabel=None,xlim=None,ylim=None,savefig=None, plt_eps=None, plt_pdf= None, fontsize=None):
	fig = plt.figure()
	plt.plot(xh[:], yh[:], 'r-', linewidth=2, markersize = 5,label=yh_label)
	legend = plt.legend(loc='best',fontsize = fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel(xlable, fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)
	plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim) 
	fig.tight_layout()
	if savefig:
		path_fig_save = savefig
		fig.savefig(path_fig_save+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(path_fig_save))
		if plt_pdf:
			plt.savefig('{}.pdf'.format(path_fig_save))
	plt.close(fig)

def plot_1Dcurve_1Dt_Sin(t,xh,yh,yh_label=None,xlable=None,ylabel=None,xlim=None,ylim=None,savefig=None, plt_eps=None, plt_pdf= None, fontsize=None):
	fig = plt.figure()
	plt.plot(xh[:], yh[:], 'r-', linewidth=2, markersize = 5,label=yh_label)
	legend = plt.legend(loc='best',fontsize = fontsize)
	if t == 0.8:
		xEx1 = np.array([[0.9, 0.94, 0.96, 0.98, 0.99, 0.999, 1.0]]).T
		cEx1 = np.array([[-0.30516, -0.42046, -0.47574, -0.52913, -0.55393, -0.26693, 0.0]]).T
		plt.plot(xEx1, cEx1,'bo',linewidth=2, markersize = 5)
	elif t == 1.0:    
		xEx2 = np.array([[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.94, 0.98, 0.99, 0.999, 1.0]]).T
		cEx2 = np.array([[0.93623, 0.98441, 0.93623, 0.79641, 0.57862, 0.30420, 0.18446, 0.06181, 0.03098, 0.00474, 0.0]]).T
		plt.plot(xEx2, cEx2,'bo',linewidth=2, markersize = 5)
	elif t == 1.6:
		xEx3 = np.array([[0.9, 0.94, 0.96, 0.98, 0.99, 0.999, 1.0]]).T
		cEx3 = np.array([[0.78894, 0.85456, 0.88237, 0.90670, 0.91578, 0.43121, 0.0]]).T
		plt.plot(xEx3, cEx3,'bo',linewidth=2, markersize = 5)

	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel(xlable, fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)
	plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim) 
	fig.tight_layout()
	if savefig:
		path_fig_save = savefig
		fig.savefig(path_fig_save+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(path_fig_save))
		if plt_pdf:
			plt.savefig('{}.pdf'.format(path_fig_save))
	plt.close(fig)

def plot_1Dcurve2(xh,yh,xref,yref,yh_label=None,yf_label=None,xlable=None,ylabel=None,xlim=None,ylim=None,savefig=None, plt_eps=None, plt_pdf= None, fontsize=None):
	fig = plt.figure()
	plt.plot(xref[:], yref[:],'b--', linewidth=2, markersize = 5, label=yf_label)
	plt.plot(xh[:], yh[:], 'r-', linewidth=2, markersize = 5,label=yh_label)
	legend = plt.legend(loc='best',fontsize = fontsize)
	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	plt.xlabel(xlable, fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)
	plt.xlim(xlim)
	if ylim:
		plt.ylim(ylim) 
	fig.tight_layout()
	if savefig:
		path_fig_save = savefig
		fig.savefig(path_fig_save+'.png',dpi=300)
		if plt_eps:
			plt.savefig('{}.eps'.format(path_fig_save))
		if plt_pdf:
			plt.savefig('{}.pdf'.format(path_fig_save))
	# fig.clf()
	plt.close(fig)

# def sub_plt_soln1D(lb,ub,Xm,U_pred,t,savefig=None, visual=None,plt_eps=None,zlim=None):
# 	def draw():
# 		nn = 200
# 		x = np.linspace(lb[0], ub[0], nn)

# 		U_plot = griddata(Xm, U_pred.flatten(), (XX, YY), method='cubic')
# 		fig = plt.figure()

# 		plt.pcolor(XX, YY, U_plot, cmap='viridis')
# 		# plt.plot(X_k[:,0], X_k[:,1], 'ko', markersize = 1.0)
# 		if zlim:
# 			plt.clim(*zlim)
# 		plt.jet()
# 		plt.colorbar()
# 		plt.xticks(fontsize=14)
# 		plt.yticks(fontsize=14)
# 		plt.xlabel('$x_1$', fontsize=16)
# 		plt.ylabel('$x_2$', fontsize=16)
# 		# plt.title('$C(x_1,x_2)$', fontsize=16)
# 		# plt.axes().set_aspect('equal')
# 		fig.tight_layout()
# 		plt.axis('equal')
# 		# path_fig_save = path_fig+'map_k_'+'_r'+str(seed_num)+'_k'+str(N_k)+'_h'+str(N_h)+'_f'+str(N_f)+'_s'+str(i_loop)+'_c'+str(N_c)+'_fc'+str(N_fc)+'_pred'
# 		# # plt.savefig(path_fig_save+'.eps', format='eps', dpi=500)
# 		# fig.savefig(path_fig_save+'.png',dpi=300)
		
# 		if savefig:
# 			path_fig_save = savefig+'_t_'+str(t)
# 			# path_fig_save = savefig
# 			fig.savefig(path_fig_save+'.png',dpi=300)
# 			if plt_eps:
# 				plt.savefig('{}.eps'.format(path_fig_save))
# 		fig.clf()
# 	if visual:
# 		drawnow(draw)
# 	else:
# 		draw()



