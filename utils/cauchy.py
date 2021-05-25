# Python file to collect any code related to Cauchy (outside sparse coding)

from autograd import numpy as np
from autograd import grad
from scipy.optimize import minimize
from scipy.stats import iqr
from matplotlib import pyplot as plt

def pdf(x, gamma, delta=0.):
	return gamma/(np.pi*(gamma**2 + (x-delta)**2))

def log_likelihood(gamma, x, delta=0.):
	return -np.sum(np.log(pdf(x, gamma, delta) + np.spacing(1)))

# From Alin's paper
def estimate_gamma(data, gamma=1., delta=0., method='mle'):
	if method == 'mle':
		return minimize(log_likelihood, gamma, args = (data, delta), method = 'Nelder-Mead').x[0]
	elif method == 'iqr':
		return iqr(data)/2

def estimate_gammas(data, gamma=1., delta=0., method='mle', mode='trial'):
	if np.size(data.shape) >2:
		if mode == 'trial':
			gammas = np.ones(data.shape[0])
			for i in range(data.shape[0]):
				gammas[i] = estimate_gamma(data[i], gamma, delta, method)
		else:
			gammas = np.ones([data.shape[0], data.shape[1]])
			for i in range(data.shape[0]):
				for j in range(data.shape[1]):
					gammas[i,j] = estimate_gamma(data[i,j].ravel(), gamma, delta, method)
		return gammas
	else:
		return estimate_gamma(data, gamma, delta, method)
	

# Used in SaprseDT paper
def estimate_alpha(Y,num_samples):
    """
    We use the estimator in Reconstruction of Ultrasound RF Echoes Modeled as Stable Random Variables
    :param Y: Y is the observations
    :param num_samples: Number of Us used for estimating alpha
    :return: alpha_hat
    """
    m = Y.shape[0]
    alphahat = 0
    count = 0
    while count <= num_samples:
        U = np.random.normal(0,1,[m,1])
        UT = np.transpose(U)
        sig = np.log(abs(np.dot(UT,Y)))
        k1 = np.mean(sig)
        k2 = np.mean((sig[:]-k1)**2)
        if k2 != 0:
            alpha_sq = (2 * np.pi**2 /(12*k2*(1 - np.pi**2/(12*k2))))
            if 0<alpha_sq<4 :
                alpha_sq = alpha_sq ** 0.5
                alphahat += alpha_sq
                count += 1
    alpha = alphahat/count
    return alpha


def plot_pdf(x, params):
	fig, ax = plt.subplots(1,1)
	for param in params:
		ax.plot(x, pdf(x, param["scale"], param["loc"]), param["style"], label=param["title"])
	ax.legend()
	plt.show()