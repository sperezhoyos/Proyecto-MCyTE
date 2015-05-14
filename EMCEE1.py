import os
import emcee
import triangle
from pylab import *
from math import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedFormatter
from numpy import *
import scipy.optimize as op
data = ()
lon = []
londism = []
iparf = []
value = []
muobs = []
mu0obs = []
mod = []
interpolated_values = []
delta = []
x = []
y = []
y1 = []
yerr = []
values = []
sigma2 = []
y1_ticks = []
y2_ticks = []

def func(x, P3, T1):
	lines1 = open('file1.txt','r')
	lines2 = open('file2.txt','r')
	lines3 = open('file3.txt','r')
	mu = []
	mu0 = []
	deltaphi = []
	for line in lines1:
		mu.append(float(line))
	for line in lines2:
		mu0.append(float(line))
	for line in lines3:
		deltaphi.append(float(line))
	newmodel = open('model.ini','w')
	newmodel.write("'LAMBDA (MICRONS) '     0.890" + "\n")
	newmodel.write("'NUMBER OF LAYERS '     -7" + "\n")
	newmodel.write("'OUTPUT SELECTION '     'points' 45" + "\n")
	newmodel.write("'PLANET           '     'saturn'" + "\n")
	newmodel.write("'P.GRAPH. LATITUDE'     +3.5" + "\n")
	newmodel.write("'MU               '     ")
	for item in mu:
		newmodel.write(str(item) + " ")
	newmodel.write("\n")
	newmodel.write("'MU0              '     ")
	for item in mu0:
		newmodel.write(str(item) + " ")
	newmodel.write("\n")
	newmodel.write("'FI-FI0 (deg)     '     ")
	for item in deltaphi:
		newmodel.write(str(item) + " ")
	newmodel.write("\n")
	newmodel.write("'-----------------'" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'gas'" + "\n")
	newmodel.write("'P_TOP (bars)     '    0.000" + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    0.0010" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'mixed'" + "\n")
	newmodel.write("'P_TOP (bars)     '    0.0010" + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    0.0100" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("'PHASE FUNCTION   '    'do_mie'" + "\n")
	newmodel.write("'TAU_EXT PARTICLES'    0.000"  + "\n")
	newmodel.write("'FOURIER TERMS    '    20" + "\n")
	newmodel.write("'Re(m) & -Im(m)   '    1.430  -0.0010" + "\n")
	newmodel.write("'SIZE DISTRIBUTION'    'hansen' 0.10 0.100" + "\n")
	newmodel.write("'RANGE rmin & rmax'    0.005  3" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'gas'" + "\n")
	newmodel.write("'P_TOP (bars)     '    0.0100"+ "\n")
	newmodel.write("'P_BOTTOM (bars)  '    "+ str(P3) + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'mixed'" + "\n")
	newmodel.write("'P_TOP (bars)     '    "+ str(P3) + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    0.500" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("'PHASE FUNCTION   '    'do_mie'" + "\n")
	newmodel.write("'TAU_EXT PARTICLES'    "+ str(T1) + "\n")
	newmodel.write("'FOURIER TERMS    '    20" + "\n")
	newmodel.write("'Re(m) & -Im(m)   '    1.430  -0.0100" + "\n")
	newmodel.write("'SIZE DISTRIBUTION'    'hansen' 1.00 0.100" + "\n")
	newmodel.write("'RANGE rmin & rmax'    0.005  5" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'gas'" + "\n")
	newmodel.write("'P_TOP (bars)     '    0.500" + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    1.2500" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'mixed'" + "\n")
	newmodel.write("'P_TOP (bars)     '    1.25000" + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    1.75000" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("'PHASE FUNCTION   '    'isotropic'" + "\n")
	newmodel.write("'OMEGA PARTICLES  '    0.99900" + "\n")
	newmodel.write("'TAU_EXT PARTICLES'    10.000" + "\n")
	newmodel.write("\n")
	newmodel.write("'TYPE OF LAYER    '     'gas'" + "\n")
	newmodel.write("'P_TOP (bars)     '    1.75000" + "\n")
	newmodel.write("'P_BOTTOM (bars)  '    6" + "\n")
	newmodel.write("'K_CH4 (1/Km-amag)'    0.0215" + "\n")
	newmodel.write("\n")
	newmodel.write("'END              '" + "\n")
	newmodel.write("\n")
	newmodel.close()	
	os.system('./atmos')	
	lines11 = open('MT3-3.5N.dat','r').readlines()[1:]#Para leer longitudes y I/F
	lines4 = open("points.dat","r").readlines()[80:]#El output del 'atmos'
	for line in lines11:
		parts = line.split()
		lon.append(float(parts[0]))
		iparf.append(float(parts[1]))
		muobs.append(float(parts[2]))
		mu0obs.append(float(parts[3]))
	i = int()
	j = int()
	k = int()
	while (i<179):
		for line in lines4:
			i += 1
			k = 1 + 4*j
			if (i == k):
				value = '0' + line.lstrip()
				mod.append(float(value))
				j += 1
	l = int()
	m = int()
	n = int()
	if (l<=177):
		for line in lines11:
			l += 1
			n = 1 + 4*m
			if (n == l):
				parts = line.split()
				londism.append(float(parts[0]))
				m += 1
	#interpolacion del modelo.
	interpolated_y = np.interp(lon,londism,mod)
	for item in interpolated_y:
		interpolated_values.append(item)	
	diferencia = iparf - interpolated_y
	for item in diferencia:
		delta.append(item)
	N = np.size(iparf)
	M = np.size(mod)
	p = int()
	for p in range(N):
		error = 0.05*interpolated_values[p]
		yerr.append(error)	
	for z in range(N):
		if x[z] == lon[z]:
			return interpolated_values[z]#Nos devuelve un valor por cada valor en x:
			return yerr[z]#Definir los valores del 'initial guess'. 
#Definimos probabilidades.
def lnlike(p, x, y, yerr):
	P3, T1 = p
	N = size(x)
	for i in range(N):
		value = ((y[i]-interpolated_values[i])**2)/(yerr[i]*yerr[i])
		values.append(value)
	return 0.5*(sum(values))

def lnprior(p):
	P3, T1 = p
	if 0.01 < P3 < 1.0 and 1.0 < T1 < 10.0:
		return 0.0
	return -np.inf

def lnprob(theta, x, y, yerr):
	lp = lnprior(theta)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnlike(theta, x, y, yerr)

P3_t = 0.08
T1_t = 2.50
lines11 = open('MT3-3.5N.dat','r').readlines()[1:]
for line in lines11:
	parts = line.split()
	x.append(float(parts[0]))
	y.append(float(parts[1]))
func(x,0.075,2.2)
ndim, nwalkers = 2, 100
pos = [0.06, 2.70]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
print 'Running MCMC...'
sampler.run_mcmc(pos, 500, rstate0=np.random.get_state())
print 'Done.'
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
fig = triangle.corner(samples, labels=["$P3$", "$T1$"], extents=[[0.01, 0.1], [1.0, 10.0]], truths=[P3_t, T1_t])
fig.set_size_inches(10,10)
fig.show()
fig.savefig("image83.png")
#Los walkers.
plt.clf()
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].axhline(P3_t, color="#888888", lw=2)
axes[0].set_ylabel("$P3$")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].axhline(T1_t, color="#888888", lw=2)
axes[1].set_ylabel("$T1$")

