import numpy as np
import matplotlib.pyplot as plt

from sys import platform

if platform == 'linux':
	plt.switch_backend('agg')

def show_imgs(imgs, title, path="", figsize=(18,9), vmax=None, vmin=None):
	vmax = imgs.max() if not vmax else vmax
	vmin = imgs.min() if not vmin else vmin
	#print(vmax, vmin)
	total = imgs.shape[0]
	if total < 6:
		rows = 1
		cols = total
		fig, axes = plt.subplots(rows, cols, figsize=figsize)
		if total == 1:
			axes.imshow(imgs[0], cmap='gray', interpolation='nearest', vmax=vmax, vmin=vmin)
			axes.set_xticks(())
			axes.set_yticks(())
		else:
			for j in range(cols):
				axes[j].imshow(imgs[j], cmap='gray', interpolation='nearest', vmax=vmax, vmin=vmin)
				axes[j].set_xticks(())
				axes[j].set_yticks(())
	else:
		[rows, cols] = get_factors(total)
		fig, axes = plt.subplots(rows, cols, figsize=figsize)
		idx = 0
		for i in range(rows):
			for j in range(cols):
				axes[i,j].set_xticks(())
				axes[i,j].set_yticks(())
				if idx < total:
					axes[i,j].imshow(imgs[idx], cmap='gray', interpolation='nearest', vmax=vmax, vmin=vmin)
				idx += 1
		#plt.tight_layout()
	plt.suptitle(title, fontsize=28)
	#plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.08)
	if path != "":
		plt.savefig(path,  bbox_inches='tight')
	else:
		plt.show()
	plt.close()

def get_factors(n):
	factors = [1]
	for i in range(2, n+1):
		if (n % i) == 0:
			factors.append(i)
	tot_factors = np.size(factors)
	if tot_factors > 2:
		middle = int(tot_factors/2)	
		if (tot_factors % 2 )== 0:
			return [factors[middle - 1], factors[middle]]
		else: 
			return [factors[middle], factors[middle]]
	else:
		return get_factors(n+1)