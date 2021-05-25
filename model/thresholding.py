# Thresholding operators
from autograd import numpy as np
from numpy.linalg import norm

from utils import math as mh

# Soft Thresholding
def soft(x, th, portion=1.):
	prox_x = np.sign(x)*np.maximum(np.abs(x) - th, 0)
	if portion < 1.:
		orig_shape = x.shape
		bottom_n = int((1.-th)*x.size)
		indxs = prox_x.ravel().argsort()[:bottom_n][::-1]
		prox_x = prox_x.ravel()
		prox_x[indxs] = 0.
		prox_x = prox_x.reshape(orig_shape)
	return prox_x

# Hard thresholding
def hard(x, th, portion=1.):
	if portion < 1.:
		orig_shape = x.shape
		bottom_n = int((1.-portion)*x.size)
		#print("portion = %s, bottom = %s" % (str(portion), str(bottom_n)))
		indxs = x.ravel().argsort()[:bottom_n][::-1]
		prox_x = x.copy().ravel()
		prox_x[indxs] = 0.
		prox_x = prox_x.reshape(orig_shape)
	else:
		prox_x = x.copy()
		prox_x[np.abs(prox_x) <=th] = 0.
	return prox_x

# l-q norm
# Python code adapted from:
# https://github.com/FWen/ncreg/
# "A survey on nonconvex regularization based sparse and low-rank
# recovery in signal processing, statistics, and machine learning,"
# IEEE Access, 2018. F. Wen, L. Chu, P. Liu, R. Qiu, 
def l_q_norm(x, lmbda, q, L=1.1):
	beta = (L/(2*lmbda*(1-q))**(1/(q-2)))
	th = lmbda*q*beta**(q-1) + L*beta - L*np.abs(x)
	#beta = (2*lmbda*(1-q))**(1/(2-q))
	#th = beta + lmbda*q*(beta**(q-1))
	idxs_low = np.where(np.abs(x)<=th)
	idxs_big = np.where(np.abs(x)>th)
	max_iter = 10
	ABSTOL = 1e-4
	prox_x = x.copy()
	x_abs = np.abs(x[idxs_big])
	y = x_abs
	for i in range(max_iter):
		deta_x = (lmbda*q*(y**(q-1)) + L*y - L*x_abs)/(lmbda*q*(q-1)*(y**(q-2)) + L)
		y = y - deta_x
		if norm(y) < ((y.size)**.5)*ABSTOL:
			i = max_iter
	y = y*np.sign(x[idxs_big])
	idxs_mid = np.where(L/2 * (y**22) - lmbda*np.abs(y)**q - L/2*(y-x[idxs_big])**2 < 0)
	y[idxs_mid] = 0.
	prox_x[idxs_low] = 0.
	prox_x[idxs_big] = y
	return prox_x

# q-shrinkage
def q_shrink(x, lmbda, q):
	thr = lmbda**(2-q)*(t**(q-1))
	return np.sign(x)*np.maximum(np.abs(x)-thr, 0)

# SCAD
def scad(x, lmbda, alpha):
	idxs_low = np.where(np.abs(x)<= 2*lmbda)[0]
	idxs_mid = np.where((np.abs(x) > 2*lmbda) &  (np.abs(x)<= alpha*lmbda))[0]
	#idxs_big = np.where(np.abs(x)> alpha*lmbda)
	prox_x = x.copy()
	prox_x[idxs_low] = np.sign(x[idxs_low])*np.maximum(np.abs(x[idxs_low])-lmbda, 0)
	prox_x[idxs_mid] = ((alpha-1)*x[idxs_mid] - np.sign(x[idxs_mid])*alpha*lmbda)/(alpha-2)
	#prox_x[idxs_low] = 0.
	return prox_x

# MCP
def mcp(x, lmbda, gamma):
	idxs_low = np.where(np.abs(x)<= lmbda)
	idxs_mid = np.where((lmbda < np.abs(x) ) & (np.abs(x) <= gamma*lmbda))
	#idxs_big = np.where(np.abs(x)> alpha*lmbda)
	prox_x = x.copy()
	prox_x[idxs_low] = 0.
	prox_x[idxs_mid] = (np.sign(x[idxs_mid])*(np.abs(x[idxs_mid])-lmbda))/(1-1/gamma)
	#prox_x[idxs_low] = 0.
	return prox_x

# Firm thresholding
def firm(x, lmbda, mu):
	idxs_low = np.where(np.abs(x)<= lmbda)
	idxs_mid = np.where(( np.abs(x) > lmbda) & (np.abs(x)<= mu))
	#idxs_big = np.where(np.abs(x)> alpha*lmbda)
	prox_x = x.copy()
	prox_x[idxs_low] = 0.
	prox_x[idxs_mid] = (np.sign(x[idxs_mid])*(np.abs(x[idxs_mid])-lmbda))*mu/(mu - lmbda)
	#prox_x[idxs_low] = 0.
	return prox_x

def log(x, lmbda, delta):
	prox_x = x.copy()
	th = (2*lmbda)**.5 - delta
	prox_x[np.abs(prox_x) <= th] = 0.
	prox_x[prox_x> th] = 0.5*((prox_x[prox_x >  th] - delta) + np.sqrt((prox_x[prox_x > th] + delta)**2-2*lmbda))
	prox_x[prox_x<-th] = 0.5*((prox_x[prox_x < -th] + delta) - np.sqrt((prox_x[prox_x <-th] - delta)**2-2*lmbda))
	return prox_x

def cauchy(x, lmbda, gamma, hard=0., verbose=False, pos_only=False):
	a = 1
	b = -x
	c = gamma**2 + lmbda
	d = -x * (gamma**2)

	# p = lmbda + gamma**2 - (x**2)/3
	# q = (-2/27)*(x**3) + (lmbda/3 -(2/3)*gamma**2)*x
	# delta = (q**2)/4 + (p**3)/27

	p = mh.get_p(a, b, c, d)
	q = mh.get_q(a, b, c, d)
	delta = mh.get_delta(p, q)

	#print("\n\n")
	#for i in range(x.size):
	#	print("%f -> %f" % (x[i], delta[i]))
	if type(x) != float:
		# Special care taken when delta is negative -- three real roots are obtained.
		idxs_neg = np.where(delta < 0)
		delta_aux = delta.astype(np.complex)
	else:
		delta_aux = delta

	delta_sqr = delta_aux ** .5

	u = mh.get_cubic_root(-q/2 + delta_sqr)
	v = mh.get_cubic_root(-q/2 - delta_sqr)

	eps = -.5 + 0.5j*np.sqrt(3)
	prox_x1 = x/3 + u + v
	prox_x2 = x/3 + eps*u + (eps**2)*v
	prox_x3 = x/3 + (eps**2)*u + eps*v

	if pos_only:
		prox_x = prox_x1.real
	else:
		#print(type(x))
		if isinstance(x, np.ndarray):
			prox_x = prox_x1.copy()
			#print("%s\t%s\t%s\t%s"  % (str(x), str(prox_x1.real), str(prox_x2.real), str(prox_x3.real)))
			prox_x[idxs_neg] = np.sign(x[idxs_neg])*np.maximum(np.maximum(np.abs(prox_x1[idxs_neg].real), np.abs(prox_x2[idxs_neg].real)), np.abs(prox_x3[idxs_neg].real))
			prox_x[ np.abs(prox_x) < 1e-10 ] = 0.
		else:
			if delta < 0:
				#print("%s\t%s\t%s\t%s"  % (str(x), str(prox_x1.real), str(prox_x2.real), str(prox_x3.real)))
				prox_x = np.sign(x)*np.maximum(np.maximum(np.abs(prox_x1.real), np.abs(prox_x2.real)), np.abs(prox_x3.real))
			else:
				prox_x = prox_x1.real
			#if np.abs(prox_x) < 1e-10:
			#	prox_x = 0.

	#prox_x[idxs_low] = 0.

	#if hard:
	#idxs_low = np.where(np.abs(x) <= hard)
	# idxs_big = np.where(np.abs(x) >  hard)
	#prox_x1[idxs_low] = prox_x2[idxs_low] = prox_x3[idxs_low] = 0.
	#print(prox_x1.real)
	if verbose:
		print("%s,%s,%s,%s,%s,%s,%s,%s" % (str(x), str(delta), str(delta_sqr), str(q), str(prox_x1), str(prox_x2), str(prox_x3), str(prox_x)))
		#print("x =         %s" % str(x))
		#print("a =         %s" % str(a))
		#print("b =         %s" % str(b))
		#print("c =         %s" % str(c))
		#print("d =         %s" % str(d))
		#print("p =         %s" % str(p))
		#print("q =         %s" % str(q))
		#print("delta =     %s" % str(delta))
		#print("delta**.5 = %s" % str(delta_sqr))
		#print("u =         %s" % str(u))
		#print("v =         %s" % str(v))
		#print("u + v =     %s" % str(u+v))
		#print("prox_x1 =   %s" % str(prox_x1))
		#print("prox_x2 =   %s" % str(prox_x2))
		#print("prox_x3 =   %s" % str(prox_x3))
		#print("prox_x  =   %s" % str(prox_x))
	return prox_x.real

	#for i in range(x.size):
		#if(delta[i] < 0):
		#print("\n\nx= %s -> delta = %s\n\tu = %s\n\tv = %s\n\txr1 = %s\n\txr2 = %s\n\txr3 = %s\n\t%s" 
		#	% (x[i], delta[i], u[i], v[i], prox_x1[i], prox_x2[i], prox_x3[i], prox_x2[i] == prox_x3[i].conjugate()))
	#return prox_x1, prox_x2, prox_x3, delta

def get_cauchy_th(gamma, lmbda):
	a = 4*gamma**2
	b = 8*gamma**4 -20*lmbda*gamma**2-lmbda**2
	c = 4*(lmbda+gamma**2)**3

	inner = b**2 - 4*a*c

	if inner >= 0:
		th1 = (-b + (inner)**.5)/(2*a)
		th2 = (-b - (inner)**.5)/(2*a)	
		#print("inner = %.12f" % (b**2 - 4*a*c))
		#print("th1 = %.12f, th2 = %.12f" % (th1, th2))
		if (th1 > 0) and (th2 > 0):
			return np.max([th1**.5, th2**.5])
		elif th1 > 0:
			return th1**.5
		elif th2 > 0:
			return th2**.5
	return 0.

def shrink_curve(x, curve):
	approximation = interp1d(curve[0], curve[1], kind='cubic')
	return approximation(x)

def shrink(x, opt, params, curve=None):
	prox = np.zeros(x.shape)
	if opt == "laplace":
		prox = soft(x, th = params["lmbda"], portion = params["portion"])
	elif opt == "hard":
		prox = hard(x, th = params["lmbda"], portion = params["portion"])
	elif opt == "q_shrink":
		prox = q_shrink(x, lmbda= params["lmbda"], q= params["q"])
	elif opt == "scad":
		prox = scad(x, lmbda= params["lmbda"], alpha= params["alpha"])
	elif opt == "mcp":
		prox = mcp(x, lmbda= params["lmbda"], gamma= params["gamma"])
	elif opt == "firm":
		prox = firm(x, lmbda= params["lmbda"], mu= params["mu"])
	elif opt == "log":
		prox = log(x, lmbda= params["lmbda"], delta= params["delta"])
	elif opt == "cauchy":
		#print(opt)
		if curve:
			return curve(x)
		else:
			prox = cauchy(x, lmbda= params["lmbda"], gamma= params["gamma"], hard = params["hard"])
	return prox