#############################################################
# Control Parameters
#############################################################
print(60*'=')
print('Defining Control Parameters')

Train_iter = 500 #10
n_rho = 25
n_thick = 6
Delta_f = 11

print('Finished')
print(60*'=')
#############################################################
# Importing Libraries
#############################################################
print(60*'=')
print('Importing Libraries')

import jax 
import jax.numpy as jnp 
import jax.random as jr
from jax import config 
import gpjax as gpx
import numpy as np
import os
import sys 
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import qmc
import optax as ox

print('Finished')
print(60*'=')
#############################################################
# Configuring JAX
#############################################################
print(60*'=')
print('Configuring JAX')

config.update("jax_enable_x64", True)

key = jr.key(42)

print('Finished')
print(60*'=')
#############################################################
# Configuring CUDA
#############################################################
print(60*'=')
print('Configuring CUDA')

print(f"JAX devices available: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Force GPU if available, else warn
if jax.default_backend() != "gpu":
    print("[WARNING] No GPU detected — running on CPU.")
# end if

print('Finished')
print(60*'=')
#############################################################
# Loading Training Data
#############################################################
print(60*'=')
print('Loading Data')

rho_values = np.linspace(0.2, 0.8, n_rho)
thick_values = np.arange(n_thick) + 1.0
freq_values = np.arange(1.0, 800.0, Delta_f)
print('rho_values = ',rho_values)
print('thick_values = ',thick_values)
print('freq_values = ',freq_values)

frf_tensor = np.zeros((len(rho_values), len(thick_values), len(freq_values)))
for cont1 in range(n_rho):
    for cont2 in range(n_thick-1):
        filn = '/work/ljk354/FRF_Optim/Learning_GP/Data_Th/Results_T1D' + str(cont1+1) + 'H' + str(2*cont2+1) + '.txt'
        frf_tensor[cont1,cont2+1,:] = np.loadtxt(filn, skiprows=3)[::Delta_f,1]
    # end for
    filn = '/work/ljk354/FRF_Optim/Learning_GP/Data/Results_T1D' + str(cont1+1) + '.txt'
    frf_tensor[cont1,0,:] = np.loadtxt(filn, skiprows=3)[::Delta_f,1]
# end for

print('Finished')
print(60*'=')
#############################################################
# Plotting Loaded Data
#############################################################
print(60*'=')
print('Plotting Loading Data')

fig, axs = plt.subplots(2,2,figsize=(16,8))
plt.subplots_adjust(hspace=0.4,wspace=0.2)
cmap = plt.cm.viridis 

ax = axs[0,0]
for cont1, rho in enumerate(rho_values):
    color = cmap(cont1/max(1,len(rho_values)-1))
    ax.plot(freq_values, np.log10(np.abs(frf_tensor[cont1,0,:]))-cont1/5, label=f'rho={rho:.2f}', color=color, lw=2)
# end for
ax.set_xlabel('Frequency $[kHz]$',fontsize=18)
ax.set_ylabel('$Log_{10}(|U|)$ $[m]$',fontsize=18)
ax.set_title('FRF for Different Densities',fontsize=22,fontweight='bold')

ax = axs[0,1]
for cont1, thick in enumerate(thick_values):
    color = cmap(cont1/max(1,len(thick_values)-1))
    ax.plot(freq_values, np.log10(np.abs(frf_tensor[0,cont1,:]))-cont1, label=f'thick={thick:.2f}', color=color, lw=2)
# end for
ax.set_xlabel('Frequency $[kHz]$',fontsize=18)
ax.set_ylabel('$Log_{10}(|U|)$ $[m]$',fontsize=18)
ax.set_title('FRF for Different Thicknesses',fontsize=22,fontweight='bold')

ax = axs[1,0]
for cont1, rho in enumerate(rho_values):
    color = cmap(cont1/max(1,len(rho_values)-1))
    ax.plot(freq_values, (-np.sign(frf_tensor[cont1,0,:])+1)*90.0-190*cont1, label=f'rho={rho:.2f}', color=color, lw=2)
# end for
ax.set_xlabel('Frequency $[kHz]$',fontsize=18)
ax.set_ylabel('Phase $[^o]$',fontsize=18)
ax.set_title('Phase for Different Densities',fontsize=22,fontweight='bold')

ax = axs[1,1]
for cont1, thick in enumerate(thick_values):
    color = cmap(cont1/max(1,len(thick_values)-1))
    ax.plot(freq_values, (-np.sign(frf_tensor[0,cont1,:])+1)*90.0-190*cont1, label=f'thick={thick:.2f}', color=color, lw=2)
# end for
ax.set_xlabel('Frequency $[kHz]$',fontsize=18)
ax.set_ylabel('Phase $[^o]$',fontsize=18)
ax.set_title('Phase for Different Thicknesses',fontsize=22,fontweight='bold')

fig.savefig('Training_Data.png', dpi=350, bbox_inches='tight')
plt.close(fig)

print('Finished')
print(60*'=')
#############################################################
# Formatting Training Data for GP
#############################################################
print(60*'=')
print('Formatting Training Data for GP')

X_Training = jnp.zeros((len(rho_values)*len(thick_values)*len(freq_values), 3))
Y_Training = jnp.zeros((len(rho_values)*len(thick_values)*len(freq_values), 1))
cont0 = 0
for cont1 in range(len(rho_values)):
    for cont2 in range(len(thick_values)):
        for cont3 in range(len(freq_values)):
            X_Training = X_Training.at[cont0,:].set(jnp.array([rho_values[cont1], thick_values[cont2], freq_values[cont3]]))
            Y_Training = Y_Training.at[cont0,0].set(frf_tensor[cont1, cont2, cont3])
            cont0 += 1
        # end for
    # end for
# end for

Dataset = gpx.Dataset(X=X_Training, y=Y_Training)
print('Dataset = ', Dataset)

print('Finished')
print(60*'=')
#############################################################
# Define Posterior
#############################################################
print(60*'=')
print('Defining Posterior')

meanf = gpx.mean_functions.Constant()
print('meanf = ', meanf)

kernel_rho = gpx.kernels.Matern52(active_dims=[0],lengthscale=jnp.array([0.1]),variance=jnp.array([1.0]))
print('kernel_rho = ', kernel_rho)

kernel_thick = gpx.kernels.Matern52(active_dims=[1],lengthscale=jnp.array([1.0]),variance=jnp.array([1.0]))
print('kernel_thick = ', kernel_thick)

kernel_freq = gpx.kernels.RationalQuadratic(active_dims=[2],lengthscale=jnp.array([10.0]),variance=jnp.array([1.0]),alpha=jnp.array([1.0]))
print('kernel_freq = ', kernel_freq)

kernel_global = gpx.kernels.ProductKernel(kernels=[kernel_rho, kernel_thick, kernel_freq])
print('kernel_global = ', kernel_global)

likelihood = gpx.likelihoods.Gaussian(num_datapoints=Dataset.n)
print('likelihood = ', likelihood)

prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel_global)
print('prior = ', prior)

posterior = prior * likelihood
print('posterior = ', posterior)

print('Finished')
print(60*'=')
#############################################################
# Create Sparse Gaussian Process Regression model
#############################################################
print(60*'=')
print('Creating Sparse Gaussian Process Regression model')

mins = jnp.amin(X_Training, axis=0)
print('mins = ', mins)
maxs = jnp.amax(X_Training, axis=0)
print('maxs = ', maxs)

sampler = qmc.Halton(d=3, scramble=True, seed=42)
print('sampler = ', sampler)
sample_points = sampler.random(n=2048)
print('sample_points = ', sample_points)
Z = mins + sample_points * (maxs - mins)
print('Z = ', Z)

print('Plotting Sampling Data')
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.bar3d(mins[0],mins[1],mins[2],maxs[0]-mins[0],maxs[1]-mins[1],maxs[2]-mins[2], alpha=0.1, color='gray')

ax.scatter(Z[:,0], Z[:,1], Z[:,2], 'o', color='blue', alpha=0.5)

fig.savefig('Sampling_Data.png', dpi=350, bbox_inches='tight')
plt.close(fig)

SGPR_model = gpx.variational_families.CollapsedVariationalGaussian(
    posterior=posterior, inducing_inputs=Z
)
print('SGPR_model = ', SGPR_model)

print('Finished')
print(60*'=')
#############################################################
# Train model
#############################################################
print(60*'=')
print('Training model with ', Train_iter, ' iterations')

opt_posterior, history = gpx.fit(
    model=SGPR_model,
    objective=lambda p, d: -gpx.objectives.collapsed_elbo(p, d),
    train_data=Dataset,
    optim=ox.adamw(learning_rate=1e-2),
    num_iters=Train_iter,
    safe=True,
    key=key,
)

print('opt_posterior = ', opt_posterior)
print('history = ', history)
print('Finished')
print(60*'=')
#############################################################
# Plotting model trainning history
#############################################################
print(60*'=')
print('Plotting model trainning history')

fig, ax = plt.subplots()

ax.plot(history, color='red', lw=2)

ax.set_xlabel('Training iterate', fontsize=18)
ax.set_ylabel('Evidence Lower Bound', fontsize=18)

fig.savefig('Trainning_History.png', dpi=350, bbox_inches='tight')
plt.close(fig)

print('Finished')
print(60*'=')
#############################################################
# Making 1 Prediction with the trained model
#############################################################
print(60*'=')
print('Making 1 Prediction with the trained model')

X_test = jnp.zeros((800, 3))
for cont1 in range(800):
    X_test = X_test.at[cont1,:].set(jnp.array([0.2, 0.0, cont1+1.0]))
# end for
print('X_test = ', X_test)

latent_dist = opt_posterior(X_test,train_data=Dataset)
print('latent_dist = ', latent_dist)

predictive_dist = posterior(X_test, train_data=Dataset)
print('predictive_dist = ', predictive_dist)

inducing_points = opt_posterior.inducing_inputs
print('inducing_points = ', inducing_points)

samples = latent_dist.sample(key=key, sample_shape=(20,))
print('samples = ', samples)

predictive_mean = predictive_dist.mean
print('predictive_mean = ', predictive_mean)

predictive_std = jnp.sqrt(predictive_dist.variance)
print('predictive_std = ', predictive_std)

print('Plotting Prediction with the trained model')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freq_values, np.log10(np.abs(frf_tensor[0,0,:])), color='red', lw=2, label='Training Data')
ax.plot(X_test[:,2], np.log10(np.abs(predictive_mean)), color='blue', lw=2, label='Predictive Mean')

# ax.fill_between(X_test[:,2], predictive_mean - 2 * predictive_std, predictive_mean + 2 * predictive_std, color='blue', alpha=0.2, label='Predictive Std Dev')

# ax.plot(X_test[:,2], np.log10(np.abs(predictive_mean + 1 * predictive_std)), color='blue', lw=1, ls='--', label='Predictive Mean - 2 Std Dev')
# ax.plot(X_test[:,2], predictive_mean - 2 * predictive_std, color='blue', lw=1, ls='--', label='Predictive Mean + 2 Std Dev')
ax.set_xlabel('Frequency $[kHz]$', fontsize=18)
ax.set_ylabel('$log_{10}(|U|)$ $[m]$', fontsize=18)

fig.savefig('Prediction.png', dpi=350, bbox_inches='tight')
plt.close(fig)

print('Finished')
print(60*'=')