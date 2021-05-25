import numpy as np

# The cubic root of some number is not well done by numpy...
def get_cubic_root(x):
	if type(x) == float:
		x_cr = np.complex(np.sign(x)*(np.abs(x)**(1/3)))
	elif type(x) == complex:
		x_cr = x**(1/3)
	else:
		x_cr = x_cr = np.zeros(x.shape, dtype=np.complex)
		x_cr[x.imag == 0] = np.sign(x[x.imag == 0].real)*(np.abs(x[x.imag == 0].real)**(1/3))
		x_cr[x.imag != 0] = x[x.imag !=0 ]**(1/3)
	return x_cr

# For Cardano's Formula
def get_p(a, b, c, d):
	p = (3*a*c-b**2)/(3*a)
	return p

def get_q(a, b, c, d):
	q = (2*(b**3)-9*a*b*c+27*(a**2)*d)/(27*(a**3))
	return q

def get_delta(p, q):
	return (q**2)/4 + (p**3)/27

# maxVal should be 1 if the data is double and 255 for 8-bit unsigned integer
def get_psnr(original, estimated, max_val=1.):
	orig_im = original[:]
	estim_im = estimated[:]
	mse_val = (1/orig_im.size) * np.sum((orig_im - estim_im)**2) 
	psnr = 10 * np.log10(max_val**2/mse_val)
	return psnr

def get_avg_psnr(originals, estimated, max_val=1.):
	n_samples = originals.shape[0]
	avg = 0
	for i in range(n_samples):
		avg += get_psnr(originals[i], estimated[i], max_val)
	avg /= n_samples
	return avg