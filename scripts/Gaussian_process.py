import matplotlib.pylab as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
from pydmd import DMD
import pickle
import time


errors = pickle.load(open('/home/rabab/Research/ROM/physor2020_transient_ROMs/data/surrogates_errors.p', 'rb'))
error_power_pod_gp = errors['power_error']['POD-GP']
snapshots = np.loadtxt('./Data/transient_sanpshots_UQ.txt')

Ncells = 350
flux = snapshots[:, :2*Ncells].T
th_flux = flux[Ncells:, :]
fast_flux = flux[:Ncells, :]

tf = 62
dt = 0.1
times = np.arange(0, tf+dt, dt)

# get POD modes
U, S, V = np.linalg.svd(flux)
U_th, S_th, V_th = np.linalg.svd(th_flux)
U_f, S_f, V_f = np.linalg.svd(fast_flux)

r = 15

Ur = U[:, :2*r]
a = Ur.T@flux

U_th = U_th[:, :r]
U_f = U_f[:, :r]

a_th = U_th.T@th_flux
a_fast = U_f.T@fast_flux

#%%
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

t1 = time.time()
## Gaussian Process
# Instantiate a Gaussian Process model
X = np.arange(0, 62.1, .1).reshape(-1, 1)
kernel = 1 * RBF(0.1, (1e-2, 1e2))

gp_th = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp_fast = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp_th.fit(X, a_th.T)
gp_fast.fit(X, a_fast.T)
gp.fit(X, a.T)


a_th_pred, sigma = gp_th.predict(X, return_std=True)
a_fast_pred, sigma = gp_fast.predict(X, return_std=True)
a_pred, sigma = gp.predict(X, return_std=True)

flux_th_pred = U_th@a_th_pred.T
flux_fast_pred = U_f@a_fast_pred.T
flux_pred = Ur@a_pred.T

#%% compute predicted powers
Ncells = 350
E = 200*1E6*1.6E-19  

nufiss = np.array([[0.000, 0.003, 0.003, 0.003, 0.003, 0.003, 0.000],
                  [0.000, 0.19, 0.19, 0.19, 0.19, 0.19, 0.000]])

fiss = np.repeat(nufiss, 50, axis=1)/2.5
power = E*np.sum(fiss[0]* flux.T[:, :Ncells] + fiss[1]* flux.T[:, Ncells:], axis=1)
power_pred = E*np.sum(fiss[0] * flux_fast_pred.T + fiss[1] * flux_th_pred.T, axis=1)
power_pred_one_basis = E*np.sum(fiss[0] * flux_pred[:350, :].T + fiss[1] * flux_pred[350: , :].T, axis=1)

errors_power_guassian_proc = abs(power - power_pred)/power *100
errors_power_guassian_proc_one_basis = abs(power - power_pred_one_basis)/power *100

#%%
plt.figure(1)
plt.semilogy(times, errors_power_guassian_proc, 'o' )
plt.semilogy(times, error_power_pod_gp, '*')
plt.xlabel('time (sec)')
plt.ylabel('relative error (%)')
plt.legend(['Garlekin-GP', 'Galerkin projection'])
plt.savefig('1D_transient.jpg')

#%%

fig = plt.figure(2)
host = fig.add_subplot(111)
par1 = host.twinx()
host.set_xlabel("time (s)")
host.set_ylabel("power (W/cm$^3$)")
par1.set_ylabel("absolute error (%)")
p1, = host.plot(times, power, color='k', label="Power")
host.ticklabel_format(style = 'scientific', axis='y', scilimits = (0,0))
p2, = par1.semilogy(times, abs(power - power_pred), '*', color='violet')
p3, = par1.semilogy(times, error_power_pod_gp*power, 'p', color='yellow')


host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
plt.savefig('1D_transient_2.jpg')


#%%
# lRA
import pickle
import scipy as sp
import plot_setting 


kappa = 3.204e-11

LRA_data = pickle.load(open('/home/rabab/Research/ROM/LRA_MO/data/inputs/diffusion2x2_ref_with_mesh_temps.p', 'rb'), encoding='latin')

t = np.array(LRA_data['times'])   # time
mp = np.array(LRA_data['meshpower']).T # mesh-dependent powers
mp = mp * kappa
p = np.array(LRA_data['power'])# total power
c = p[0]/sp.sum(mp,axis=0)[0] # use to reconstruct power from mesh power

#plt.semilogy(t, p, 'k-', label='reference')


fuel_idx = mp[:, 0]>0                  # pick out fuel mesh
tmp_reduced = mp[fuel_idx, :] # extract fuel data
tmp_full = 0*mp[:, :]  # initialize full data


U, S, V = np.linalg.svd(tmp_reduced)
r = 20
Ur = U[:, :r]

# compute temporal coefficient
a = Ur.T@tmp_reduced
a_predict = np.zeros(a.shape)
kernel = C(10.0, (1e-3, 1e3)) * RBF(0.5, (1e-2, 1))
kernel = 10 * RBF(1, (1e-2, 1))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
X = t.reshape(-1, 1)
y = a.copy()
gp.fit(X, y.T)

a_pred, sigma = gp.predict(X, return_std=True)
 
tmp_reduced_pred = Ur@a_pred.T
tmp_full[fuel_idx] = tmp_reduced_pred

p_pred = c*sp.sum(tmp_full, axis=0)

err = abs(p - p_pred) *100
#%%          
fig = plt.figure(3)
host = fig.add_subplot(111)
par1 = host.twinx()
host.set_xlabel("time (s)")
host.set_ylabel("power (W/cm$^3$)")
par1.set_ylabel("absolute error (%)")
p1, = host.plot(t, p, color='blue', label="Power")
host.ticklabel_format(style = 'scientific', axis='y', scilimits = (0,0))
p2, = par1.semilogy(t, err, '*', color='violet')
host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
plt.savefig('LRA.jpg')
