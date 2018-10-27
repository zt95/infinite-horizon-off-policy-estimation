import numpy as np
from scipy.optimize import linprog
import quadprog

# Want to min_w max_f w^T M f
# with w >= 0 and \sum w = 1

def linear_solver(n, M):
	M -= np.amin(M)	# Let zero sum game at least with nonnegative payoff
	c = np.ones((n))
	b = np.ones((n))
	res = linprog(-c, A_ub = M.T, b_ub = b)
	w = res.x
	return w/np.sum(w)

def quadratic_solver(n, M):
	qp_G = np.matmul(M, M.T)
	qp_a = np.zeros(n, dtype = np.float64)
	qp_C = np.zeros((n,n+1), dtype = np.float64)
	for i in range(n):
		qp_C[i,0] = 1
		qp_C[i,i+1] = 1
	qp_b = np.zeros(n+1, dtype = np.float64)
	qp_b[0] = 1
	meq = 1
	res = quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)
	w = res[0]
	return w

n = 10
M = (np.random.rand(n,n) - 0.5) * (1.0 + np.random.rand()) * 10.0
w1 = linear_solver(n, M)
w2 = quadratic_solver(n, M)

print 'Linear result:'
print w1
print 'Quadratic result:'
print w2