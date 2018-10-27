import numpy as np
from Density_Ratio_discrete import Density_Ratio_discrete
from Q_learning import Q_learning
from environment import random_walk_2d, taxi
import numpy as np
import matplotlib
font = {'family' : 'normal',
        'size'   : 28}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def roll_out(state_num, env, policy, num_trajectory, truncate_size):
	SASR = []
	total_reward = 0.0
	frequency = np.zeros([state_num], dtype = np.float32)
	for i_trajectory in range(num_trajectory):
		state = env.reset()
		sasr = []
		for i_t in range(truncate_size):
			#env.render()
			p_action = policy[state, :]
			action = np.random.choice(p_action.shape[0], 1, p = p_action)[0]
			next_state, reward = env.step(action)

			sasr.append((state, action, next_state, reward))
			frequency[state] += 1
			total_reward += reward
			#print env.state_decoding(state)
			#a = input()

			state = next_state
		SASR.append(sasr)
	return SASR, frequency, total_reward/(num_trajectory * truncate_size)

def KL_divergence(a, b):
	a = np.asarray(a, dtype=np.float)
	b = np.asarray(b, dtype=np.float)
	a = a/np.sum(a)		# Make sure normalized
	b = b/np.sum(b)
	return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def TV_distance(a, b):
	return 0.5 * np.sum(np.abs(a-b))

def weighted_TV_distance(a, b, weight):
	a_weight = a*weight
	a_weight /= np.sum(a_weight)
	b_weight = b*weight
	b_weight /= np.sum(b_weight)
	return 0.5 * np.sum(np.abs(a_weight - b_weight))

def learning_curve(SASR, policy0, policy1, den_discrete, f1, X):
	weighted_TV = []
	weighted_TV01 = []
	weighted_TV1 = []
	weighted_TV10 = []
	KL = []
	i = 0
	f1 = np.array(f1, dtype = np.float32)/np.sum(f1)
	for sasr in SASR:
		for state, action, next_state, _ in sasr:
			policy_ratio = policy1[state, action]/policy0[state, action]
			den_discrete.feed_data(state, next_state, policy_ratio)
		if i in X:
			x, w = den_discrete.density_ratio_estimate()
			x01, _ = den_discrete.density_ratio_estimate(regularizer = 0.1)
			x1, _ = den_discrete.density_ratio_estimate(regularizer = 1.0)
			x10, _ = den_discrete.density_ratio_estimate(regularizer = 10.0)
			weight = np.sign(w)
			t = weighted_TV_distance(x, f1, weight)
			t01 = weighted_TV_distance(x01, f1, weight)
			t10 = weighted_TV_distance(x10, f1, weight)
			t1 = weighted_TV_distance(x1, f1, weight)
			t_KL = KL_divergence(x+1e-15, f1+1e-15)
			#t2 = TV_distance(x, f1)
			print('-----Iteration {}-----'.format(i))
			print('reg = {}, weighted_TV_distance = {}'.format(0.01, t))
			print('reg = {}, weighted_TV_distance = {}'.format(0.1, t01))
			print('reg = {}, weighted_TV_distance = {}'.format(1., t1))
			print('reg = {}, weighted_TV_distance = {}'.format(10., t10))
			print 'KL_divergence = {}'.format(t_KL)
			print('---------------')
			weighted_TV.append(t)
			weighted_TV01.append(t01)
			weighted_TV10.append(t10)
			weighted_TV1.append(t1)
			KL.append(t_KL)
		i += 1
	ax = plt.subplot()
	plt.plot(X, weighted_TV, color = (0.1974, 0.5129, 0.7403), lw = 4.0, marker = 'd', ms = 10.0)
	ax.set_ylim(ymin = 0.0, ymax = 0.4)

	plot_filename = 'taxi_learning_curve.pdf'
	ppPDF = PdfPages(plot_filename)
	ppPDF.savefig()
	ppPDF.close()
	x, w = den_discrete.density_ratio_estimate()
	return w, x


def learning_scatter(d_pi, d_hat, w):
	plt.clf()
	weight = np.sign(w)
	d_pi *= weight
	d_pi /= np.sum(d_pi)

	d_hat *= weight
	d_hat /= np.sum(d_hat)
	ax = plt.subplot()
	plt.scatter(d_pi, d_hat, s = 45.0, alpha = 0.5, c = (0.3686, 0.3098, 0.6353))
	plt.plot(np.arange(0, 0.03, 0.0001), np.arange(0, 0.03, 0.0001), linestyle = '-.', color = (1.0, 0.0, 0.0), lw = 0.5)
	ax.set_xlim(xmin = 0.0, xmax = 0.03)
	ax.set_ylim(ymin = 0.0, ymax = 0.03)
	plt.xticks([0.0, 0.01, 0.02, 0.03])
	plt.yticks([0.0, 0.01, 0.02, 0.03])

	plot_filename = 'taxi_learned_scatter.pdf'
	ppPDF = PdfPages(plot_filename)
	ppPDF.savefig()
	ppPDF.close()


if __name__ == '__main__':
	d_pi = np.load('exp_result/true_density.npy')
	d_hat = np.load('exp_result/estimate_d_pi.npy')
	w = np.load('exp_result/density.npy')
	learning_scatter(d_pi, d_hat, w)
	quit()
	length = 5
	NT = [30, 50, 100, 200, 400, 600, 800]
	TS = [50, 100, 200, 600, 1000, 1500]
	BP = [0, 1, 2, 3]

	num_trajectory = 600
	truncate_size = 400

	pi = np.load('taxi-policy/pi19.npy')

	env = taxi(length)
	n_state = env.n_state

	den_discrete = Density_Ratio_discrete(n_state)
	bp = 4
	SASR = np.load('temp/SASR_pi{}_seed={}.npy'.format(bp, 0))
	SASR = SASR[:num_trajectory, :truncate_size]


	SASR3 = np.load('temp/SASR_pi5_seed={}.npy'.format(0))
	SASR3 = SASR3[:, :truncate_size]

	f3 = np.zeros([n_state], dtype = np.float32)
	for state in SASR3[:,:,0].flat:
		f3[state] += 1.0

	f3 = f3/np.sum(f3)

	X = range(0,num_trajectory,10)
	pi0 = np.load('taxi-policy/pi{}.npy'.format(14+bp))
	
	w, x = learning_curve(SASR, pi0, pi, den_discrete, f3, X)
	np.save('exp_result/estimate_d_pi.npy', x)
	np.save('exp_result/density.npy', w)
	np.save('exp_result/true_density', f3)

	learning_scatter(f3, x, w)

	#np.save('exp_result/learning_curve.npy', np.vstack((X, weighted_TV)))