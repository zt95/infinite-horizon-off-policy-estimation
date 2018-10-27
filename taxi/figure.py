from __future__ import print_function
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 28}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def mean_bootstrap(A):
	N = A.shape[1]
	samples = np.random.choice(N, N, replace = True)
	return np.mean(A[:,samples],axis = 1)

def log_std(A):
	MEAN_bootstrap = mean_bootstrap(A)[:,None]
	for i in range(1000):
		temp = mean_bootstrap(A)
		MEAN_bootstrap = np.concatenate((MEAN_bootstrap, temp[:,None]), axis = 1)
	MEAN_bootstrap = np.sort(MEAN_bootstrap, axis = 1)
	return MEAN_bootstrap[:,500], MEAN_bootstrap[:,50], MEAN_bootstrap[:,950]

if __name__ == '__main__':
	mode = 'MR'
	label_name = ['On-Policy', 'Density-Ratio', 'Naive-Average', 'IS-Trajectorywise', 'IS-Stepwise', 'WIS-Trajectorywise', 'WIS-Stepwise', 'Model-Based']
	lineWidth = 4.0
	markerSize = 20.0
	elineWidth = 1.0
	color_map2 = [(0.,0.,1.),(0.,0.5,0.),(1.,0.,0.),(0.75, 0.75, 0),(0.,0.75,0.75),(0.75,0.,0.75),(0.,0.,0.),(0.3770,0.2356,0.1500)]
	color_map = [(0.3016, 0.3016, 0.3016), (1, 0, 0), (0.1974, 0.5129, 0.7403), (0.75, 0.75, 0),(0.,0.75,0.75), (0.3686, 0.3098, 0.6353), (0.3016, 0.6508, 0.4000), (0.3770,0.2356,0.1500)]
	marker_shape = ['x', 'o', 'v', 'h', 'p', 's', 'd', '^']

	if mode == 'NT':
		X = [30, 50, 100, 200, 400, 600, 800] # NT
	elif mode == 'TS':
		X = [50, 100, 200, 400, 600, 1000, 1500] # TS
	elif mode == 'BP':
		X = [0,1,2,3,4]
	elif mode == 'GM':
		X = [1.0, 0.995, 0.99, 0.98, 0.97, 0.95]
	elif mode == 'MR':
		#X = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
		X = [0.0, 0.2, 0.4, 0.6, 0.8]

	num_trajectory = 200
	truncate_size = 400
	behaviorID = 4
	gamma = 0.99
	Y = []
	for x in X:
		if mode == 'NT':
			num_trajectory = x
		elif mode == 'TS':
			truncate_size = x
		elif mode == 'BP':
			behaviorID = x
		elif mode == 'GM':
			gamma = x
		elif mode == 'MR':
			alpha = x
			if alpha == 0.0:
				y = np.load('reward_result_discounted/NT=200_TS=400_BP=4_GM=0.99.npy')
				# y = np.load('reward_result/NT=100_TS=400_BP=4.npy')
			else:
				y = np.load('reward_result_discounted/mixed_ratio={}.npy'.format(alpha))
				# y = np.load('reward_result/mixed_ratio={}.npy'.format(alpha))
		if mode != 'MR':
			y = np.load('reward_result_discounted/NT={}_TS={}_BP={}_GM={}.npy'.format(num_trajectory, truncate_size, behaviorID, gamma))
			# y = np.load('reward_result/NT={}_TS={}_BP={}.npy'.format(num_trajectory, truncate_size, behaviorID))
		Y.append(y)
	Y = np.array(Y)

	if mode == 'NT' or mode == 'BP' or mode == 'MR':
		True_reward = np.mean(Y[-1,0,:])
		MSE = np.log((Y - True_reward)**2 + 1e-20)
	elif mode == 'GM' or mode == 'TS':
		True_reward = np.mean(Y[:,0,:], axis = -1)
		MSE = np.log((Y - True_reward[:,None, None])**2 + 1e-20)

	ax = plt.subplot()
	X = np.array(X)
	if mode == 'NT':
		ax.set_xscale("log", nonposx='clip')
	'''
	if mode == 'NT':
		plt.xlabel('sample size')
	elif mode == 'TS':
		plt.xlabel('truncate size')
	else:
		plt.xlabel('closeness to target')
	plt.ylabel('log MSE')
	'''
	

	# print("Y.shape = {}".format(Y.shape))
	I = [0, 1, 2, 5, 6, 7]
	with open('taxi_discounted_data_{}.txt'.format(mode), 'w') as data_file:
	# with open('taxi_data_{}.txt'.format(mode), 'w') as data_file:
		for i in I:
			t, t_min, t_max = log_std(MSE[:,i,:])
			for j in range(len(X)):
				point = 'x {} y {} y_min {} y_max {} alg {}'.format(X[j], t[j], t_min[j], t_max[j], label_name[i])
				print(point, file = data_file)
			plt.errorbar(X, t, yerr = np.array([t_max - t, t - t_min]), color = color_map[i], lw = lineWidth, elinewidth = elineWidth, marker = marker_shape[i], markerfacecolor='None', markeredgecolor = color_map[i], markeredgewidth = lineWidth, ms = markerSize, label = label_name[i])
	#plt.legend()

	if mode == 'NT':
		ax.set_xlim(xmin = 25, xmax = 810)
		ax.set_xticks([30, 50, 100, 200, 400, 800])
		ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	elif mode == 'TS':
		ax.set_xlim(xmin = 45, xmax = 1550)
		plt.xticks([200, 400, 600, 1000, 1500])
	elif mode == 'BP':
		ax.set_xlim(xmin = 0, xmax = 4)
		plt.xticks(X)
	elif mode == 'GM':
		ax.set_xlim(xmin = 0.945, xmax = 1.005)
		plt.xticks([0.95, 0.97, 0.98, 0.99, 1])
	elif mode == 'MR':
		ax.set_xlim(xmin = -0.02, xmax = 0.82)
		plt.xticks(X)
	'''
	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	#plt.show()
	'''

	plot_filename = 'taxi_discounted_plot_{}.pdf'.format(mode)
	# plot_filename = 'taxi_plot_{}.pdf'.format(mode)
	ppPDF = PdfPages(plot_filename)
	ppPDF.savefig()
	ppPDF.close()
	