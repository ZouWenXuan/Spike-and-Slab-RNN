# =============================================================================
# Task 1: Verify the initialized recurrent neural tangent kernel
# =============================================================================

import numpy as np
import SaS_RNTK


#%%% Test
alpha = 0.5
n_in = 1
n_rec = 2000
n_out = 1
sigma_in = sigma_rec = sigma_out = sigma_h0 = 1
b_size = 2
t_max = 2
SaS_output = True
def x_func(phi):
    if t_max == 2: 
        x1 = np.array([1, -1], ndmin=2).T
        x2 = np.array([np.cos(phi), np.sin(phi)], ndmin=2).T
    if t_max == 3:
        x1 = np.array([1, -1, 1], ndmin=2).T
        x2 = np.array([np.cos(phi), np.sin(phi), -np.cos(phi)], ndmin=2).T
    x = np.concatenate((x1, x2), axis=1)
    x = np.expand_dims(x, 1)
    return x

# rntk vs phi
samples = 100
analy_rntk = np.zeros((41, b_size, b_size))
bptt_rntk = np.zeros((41, samples, b_size, b_size))

# compute
print("Compute RNTK: alpha = {}, T = {}.".format(alpha, t_max))
for i, phi in enumerate(np.arange(0, 2*np.pi + 0.05*np.pi, 0.05*np.pi)):
    print("{}. Phi = {:.4f}".format(i, phi))

    # set input
    x = x_func(phi)
    
    # analytical
    kernel1 = SaS_RNTK.RNTK(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    kernel1.GP_kernel(x, alpha)
    kernel1.compute_rntk(x, alpha, SaS_output)  
    rntk2 = kernel1.rntk
    analy_rntk[i] = 1*rntk2
    print("   Analytical finished!")
    
    # bptt
    for r in range(samples):
        network1 = SaS_RNTK.RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
        if SaS_output:
            network1.SaS_output()
        network1.gradient_bptt(x, alpha, SaS_output=SaS_output)
        rntk1 = network1.rntk_bptt(SaS_output)
        bptt_rntk[i][r] = 1*rntk1
        print("\r   BPTT samples:{}/{}".format(r+1, samples), end='')
    print("\n   BPTT finished!")

# save data
rntk_dict = {}
rntk_dict['analy'] = analy_rntk*1
rntk_dict['bptt'] = bptt_rntk*1
np.save("rntk_alpha_{}_{}_SaSop.npy".format(alpha, t_max), rntk_dict)