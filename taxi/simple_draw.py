import numpy as np
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
	return MEAN_bootstrap[:,501], MEAN_bootstrap[:,50], MEAN_bootstrap[:,950]

lineWidth = 2.0
markerSize = 8.0
elineWidth = 1.0
fontSize = 10.0

label_name = ['On policy', 'Density Ratio', 'IS-trajectorywise', 'IS-stepwise', 'WIS-trajectorywise', 'WIS-stepwise', 'Model Based', 'Naive Average']
True_reward = -0.14118925
#True_reward = [-0.197548, -0.15987925, -0.14118925, -0.1341735, -0.129886, -0.1268788 ]
#True_reward = np.array(True_reward).reshape([-1,1,1])
#X = [100, 200, 400, 600, 1000, 1500]
X = [0,1,2]
num_trajectory = 400
truncate_size = 400
behaviorID = 2
Y = []
for behaviorID in X:
	y = np.load('reward_result/NT={}_TS={}_BP={}.npy'.format(num_trajectory, truncate_size, behaviorID))
	Y.append(y)
Y = np.array(Y)


MSE = np.log((Y - True_reward)**2 + 1e-20)	#log(MSE)

X = np.array(X)
plt.xlabel('policy ID')
plt.ylabel('log MSE')
color_map = [(0.,0.,1.),(0.,0.5,0.),(1.,0.,0.),(0.75, 0.75, 0),(0.,0.75,0.75),(0.75,0.,0.75),(0.,0.,0.),(0.3770,0.2356,0.1500)]
marker_shape = ['*', 'h', 'v', '^', '>', '<', 's', 'p']

ax = plt.subplot()
#ax.set_xscale("log", nonposx='clip')
for i in range(Y.shape[1]):
	t, t_min, t_max = log_std(MSE[:,i,:])
	plt.errorbar(X, t, yerr = np.array([t_max - t, t - t_min]), color = color_map[i], lw = lineWidth, elinewidth = elineWidth, marker = marker_shape[i], ms = markerSize, label = label_name[i])

#plt.legend()


#ax.set_xlim(xmin = 90, xmax = 1600)
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.show()
'''

LC = np.load('exp_result/learning_curve.npy')
X = LC[0]
weighted_TV = LC[1]
plt.xlabel('sample size')
plt.ylabel('weighted TV distance')
plt.plot(X, weighted_TV, lw = lineWidth , ms = markerSize)
'''
plot_filename = 'plot_BP.pdf'
ppPDF = PdfPages(plot_filename)
ppPDF.savefig()
ppPDF.close()
