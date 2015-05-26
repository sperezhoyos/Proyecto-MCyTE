# Import block - clean if not needed
import os
import emcee
import triangle
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator




def read_scattering_angles():
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
        return mu, mu0, deltaphi

def write_model(mu, mu0, deltaphi, P3,T2):
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
	newmodel.write("'TAU_EXT PARTICLES'    "+ str(T2) + "\n")
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
	newmodel.write("'K_CH4 (1/Km-amag)'    21.5" + "\n")
	newmodel.write("\n")
	newmodel.write("'END              '" + "\n")
	newmodel.write("\n")
	newmodel.close()	
        return None

def execute_atmos():
	os.system('./atmos')	
        return None

def read_obs():
        lon = []
        iparf = []
        mu0obs = []
        muobs = []
	lines11 = open('MT3-3.5N.dat','r').readlines()[1:]#Para leer longitudes y I/F
	for line in lines11:
		parts = line.split()
		lon.append(float(parts[0]))
		iparf.append(float(parts[1]))
		muobs.append(float(parts[2]))
		mu0obs.append(float(parts[3]))
        return lon, iparf, muobs, mu0obs

def read_pointsdat():
        mod = []
        londism = []
	lines4 = open("points.dat","r").readlines()[80:]#El output del 'atmos'
	lines11 = open('MT3-3.5N.dat','r').readlines()[1:]#Para leer longitudes y I/F
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
        return mod,londism

def interpolate_model(lon,londism,mod,iparf):
        interpolated_values = []
        delta = []
	#interpolacion del modelo.
	interpolated_y = np.interp(lon,londism,mod)
	for item in interpolated_y:
		interpolated_values.append(item)	
	diferencia = iparf - interpolated_y
	for item in diferencia:
		delta.append(item)
#	for z in range(np.size(x)):
#		if x[z] == lon[z]:
#			return interpolated_values[z]#Nos devuelve un valor por cada valor en x:
        return interpolated_values, delta

def move_files(counter):
        global counter
        order1 = 'mv points.dat points' + str(counter,'%3g') + '.dat'
        order2 = 'mv model.ini model' + str(counter,'%3g') + '.ini'
        os.system(order1)
        os.system(order2)
        counter += 1
        return None


def func(x, P3, T1):
	mu, mu0, deltaphi = read_scattering_angles()
	write_model(mu, mu0, deltaphi, P3,T1)
	execute_atmos()
        lon, iparf, muobs, mu0obs = read_obs()
        mod,londism = read_pointsdat()
        interpolated_values, delta = interpolate_model(lon,londism,mod,iparf)
        return interpolated_values

# Define the probability function as likelihood  prior.
def lnprior(theta):
	P3, T1, lnf = theta
	if (0.01 < P3 < 1.0 and 1.0 < T1 < 10.0 and -10.0 < lnf < 1.0):
		return 0.0
	return -np.inf

def lnlike(theta, x, y, yerr):
        values = []
	P3, T1, lnf = theta
        interpolated_values = func(x,P3,T1)
	N = np.size(interpolated_values)
	for i in range(N):
		value = ((y[i]-interpolated_values[i])**2)/(2*lnf)
		values.append(value)
	return 0.5*(sum(values))

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# Main block
if __name__ == '__main__':
   # Choose the "true" parameters.
   T1_true = -0.9594
   P3_true = 4.294
   f_true = 0.534

   # Generate some synthetic data from the model.
   NN = 50
   x = []
   y = []
   lines11 = open('MT3-3.5N.dat','r').readlines()[1:]
   for line in lines11:
   	parts = line.split()
	x.append(float(parts[0]))
   yerr = 0.1+0.5*np.random.rand(NN)
   interpolated_values = func(x,0.079,5.3)
   for item in interpolated_values:
	y.append(item)
   #y += np.abs(f_true*y) * np.random.randn(NN)
   #y += yerr * np.random.randn(NN)

   print np.size(x)
   print np.size(y)
   print np.size(interpolated_values)

# Plot the dataset and the true model.
#xl = np.array([0, 10])
#plt.errorbar(x, y, yerr=yerr, fmt=".k")
#plt.xlabel("$longitud$")
#plt.ylabel("$I/F$")
#plt.tight_layout()
#plt.savefig("line-data.png")

# Do the least-squares fit and compute the uncertainties.
#A = np.vstack((np.ones_like(x), x)).T
#C = np.diag(yerr * yerr)
#cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
#P3_ls, T1_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

# Plot the least-squares result.
#pl.plot(xl, m_ls*xl+b_ls, "--k")
#pl.savefig("line-least-squares.png")



   # Find the maximum likelihood value.
   #chi2 = lambda *args: -2 * lnlike(*args)
   #result = op.minimize(chi2, [P3_true, T1_true, np.log(f_true)], args=(x, y, yerr))
   #P3_ml, T1_ml, lnf_ml = result["x"]

   # Plot the maximum likelihood result.
   #pl.plot(xl, m_ml*xl+b_ml, "k", lw=2)
   #pl.savefig("line-max-likelihood.png")
   initial_position = [0.08,5.0,0.534]
   # Set up the sampler.
   ndim, nwalkers = 3, 50
   pos = [initial_position + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
   sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

   # Clear and run the production chain.
   print("Running MCMC...")
   sampler.run_mcmc(pos, 100, rstate0=np.random.get_state())
   print("Done.")

   plt.clf()
   fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
   axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
   axes[0].yaxis.set_major_locator(MaxNLocator(5))
   axes[0].axhline(P3_true, color="#888888", lw=2)
   axes[0].set_ylabel("$P3$")

   axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
   axes[1].yaxis.set_major_locator(MaxNLocator(5))
   axes[1].axhline(T1_true, color="#888888", lw=2)
   axes[1].set_ylabel("$T1$")

   axes[2].plot(np.exp(sampler.chain[:, :, 2]).T, color="k", alpha=0.4)
   axes[2].yaxis.set_major_locator(MaxNLocator(5))  
   axes[2].axhline(f_true, color="#888888", lw=2)
   axes[2].set_ylabel("$f$")
   axes[2].set_xlabel("step number")

   fig.tight_layout(h_pad=0.0)
   fig.savefig("line-time3.png")

   # Make the triangle plot.
   burnin = 50
   samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

   fig = triangle.corner(samples, labels=["$P3$", "$T1$", "$\ln\,f$"],
                      truths=[P3_true, T1_true, np.log(f_true)])
   fig.savefig("line-triangle3.png")
