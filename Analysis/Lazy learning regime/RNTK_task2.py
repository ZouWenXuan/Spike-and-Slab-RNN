# =============================================================================
# Task 2: Learning dynamics
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import SaS_RNTK
import time


#%% BPTT & RNTK dynamics
# BPTT dynamics
def BPTT_dynamics(net, eta, learning):
    # train & test
    if learning:
        net.run_bptt(X, Y, alpha, eta, learning=learning)      
    else:
        net.run_bptt(x, y, alpha, eta, learning=learning)   
    f_ = net.y
    return f_  


# RNTK dynamics
def RNTK_dynamics(fX0, fx0, epoch, eta, dynamics_type):
    # data concatenate
    X_x = np.concatenate((X,x), axis=2)
    
    # kernel
    kernel = SaS_RNTK.RNTK(n_in, n_rec, n_out, sigma_in,\
                           sigma_rec, sigma_out, sigma_h0)
    kernel.GP_kernel_dynamics(X_x)
    kernel.compute_rntk_dynamics(X_x)  
    rntk = kernel.rntk

    # segment
    rntk_XX = rntk[0:train_size, 0:train_size]*1
    rntk_xX = rntk[train_size:(train_size+test_size), 0:train_size]*1

    # train & test dynamics
    fX = np.zeros((epoch, train_size))
    fx = np.zeros((epoch, test_size))
    # continuous
    if dynamics_type == "continuous":
        Lambda, U = np.linalg.eig(rntk_XX)
        exp_kernel = lambda t: np.dot( np.dot(U, np.diag(np.exp(-eta*Lambda*t)) ),U.T)
        inverse_kernel = np.dot( np.dot(U, np.diag(1/Lambda) ), U.T )
        for i in range(epoch):
            fX[i] = (np.dot(exp_kernel(i), (fX0 - Y).T) + Y.T).reshape(-1)
            fx[i] = (fx0.T - np.linalg.multi_dot([rntk_xX, inverse_kernel,\
                                                (np.identity(train_size) - exp_kernel(i)),\
                                                (fX0 - Y).T]) ).reshape(-1)
    # discrete  
    if dynamics_type == "discrete":       
        fX[0] = fX0*1
        fx[0] = fx0*1
        for i in range(epoch-1):
            fX[i+1] = fX[i] - eta*(np.dot(rntk_XX, (fX[i:i+1]-Y).T)).reshape(-1)
            fx[i+1] = fx[i] - eta*(np.dot(rntk_xX, (fX[i:i+1]-Y).T)).reshape(-1)
    return fX, fx, rntk




#%% Main Test
# parameter
alpha = 1
n_in, n_out = 28, 1
t_max = 28
eta = 0.001
epoch = 100
sigma_in = 1
sigma_rec = 1
sigma_out = 1
sigma_h0 = 1

# # load data
train_size, test_size = 512, 128
data_01 = np.load("mnist_data01.npy")
random_index = np.random.permutation(data_01.shape[0])
train_01 = (data_01[random_index])[0:train_size]
test_01 = (data_01[random_index])[train_size:(train_size + test_size)]
X = (train_01[:,1:].T).reshape(t_max, n_in, train_size)
Y = train_01[:,0:1].T
x = (test_01[:,1:].T).reshape(t_max, n_in, test_size)
y = test_01[:,0:1].T



# error and loss function
def error_accuracy(fx, y):
    # fx: [epoch, size], y: [1, size]
    error = 1/2*((fx-y)**2).sum(1)
    fx_hat = np.where(fx>0.5, 1, 0)
    accuracy = (fx_hat==y).mean(1)
    return error, accuracy

def parameter_dict():
    pdict = {}
    pdict['m_in'] = network1.m_in_tr*1
    pdict['m_rec'] = network1.m_rec_tr*1
    pdict['m_out'] = network1.m_out_tr*1
    pdict['pi_rec'] = network1.pi_rec_tr*1
    pdict['xi_rec'] = network1.xi_rec_tr*1
    return pdict

#%%% initialization
n_rec = 64
m_in_tr = np.random.normal(0,1,[n_rec, n_in])
m_rec_tr = np.random.normal(0,1,[n_rec, n_rec])
xi_rec_tr = np.random.random((n_rec, n_rec))
m_out_tr = np.random.normal(0,1,[n_out, n_rec])

# initial loss
network0 = SaS_RNTK.RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
network0.m_in_tr = m_in_tr*1
network0.m_rec_tr = m_rec_tr*1
network0.xi_rec_tr = xi_rec_tr*1
network0.m_out_tr = m_out_tr*1

fx0 = BPTT_dynamics(network0, eta, False)
fX0 = BPTT_dynamics(network0, eta, True)
L0, _ = error_accuracy(fX0, Y)


#%%% BPTT: 1-trial test
# initializatized kernel BPTT
t1 = time.time()
network1 = SaS_RNTK.RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
network1.m_in_tr = m_in_tr*1
network1.m_rec_tr = m_rec_tr*1
network1.xi_rec_tr = xi_rec_tr*1
network1.m_out_tr = m_out_tr*1
network1.gradient_bptt(x, 1)
kernels_bptt0 = network1.rntk_bptt()
kernels_bptt0 = (kernels_bptt0 + kernels_bptt0.T)-np.diag(np.diag(kernels_bptt0))
t2 = time.time()
print("\nCompute kernel time cost: {}s".format(t2-t1))

# parameter assign
network1 = SaS_RNTK.RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
network1.m_in_tr = network0.m_in_tr*1
network1.m_rec_tr = network0.m_rec_tr*1
network1.xi_rec_tr = network0.xi_rec_tr*1
network1.m_out_tr = network0.m_out_tr*1
network1.pi_rec_tr = network0.pi_rec_tr*1

# test
t1 = time.time()
fX_bptt = np.zeros((epoch, train_size))
fx_bptt = np.zeros((epoch, test_size))
fX_bptt[0] = fX0*1
fx_bptt[0] = fx0*1

for i in range(1, epoch):
    fx = BPTT_dynamics(network1, eta, False)
    fX = BPTT_dynamics(network1, eta, True) 
    fX_bptt[i] = fX.reshape(-1)
    fx_bptt[i] = fx.reshape(-1)
    Loss = 1/2*((fX-Y)**2).sum()
    fX_hat = np.where(fX>0.5, 1, 0)
    accuracy = (fX_hat==Y).mean()
    print("BPTT: {}, Loss={:.4f}, accuracy={:.4f}.".format(i+1, Loss/512, accuracy))
t2 = time.time()
print("\ntime cost: {}s".format(t2-t1))


#%%% RNTK: 1-trial test
fX_rntk, fx_rntk, rntk = RNTK_dynamics(fX0, fx0, epoch, eta, "continuous")
rntk_xx = rntk[512:640,512:640]
network1.gradient_bptt(x, 1)
kernels_bptt = network1.rntk_bptt()
kernels_bptt = (kernels_bptt + kernels_bptt.T)-np.diag(np.diag(kernels_bptt))

numer = np.sqrt( ((kernels_bptt - kernels_bptt0)**2).sum() )
deno = np.sqrt( (kernels_bptt0**2).sum() )
print("kernel variation: {}".format(numer/deno))


#%%% Comparison 1: bptt & rntk
# loss
Ltr_bptt, _ = error_accuracy(fX_bptt, Y)
Lte_bptt, _ = error_accuracy(fx_bptt, y)
Ltr_rntk, _ = error_accuracy(fX_rntk, Y)
Lte_rntk, _ = error_accuracy(fx_rntk, y)

plt.figure(figsize=(10, 4), dpi=300)
ax = plt.subplot(121)
step = np.arange(0,epoch,1) + 1
plt.semilogx(step, Ltr_bptt/train_size, label='BPTT', c='darkblue', lw=2, base=2)
plt.semilogx(step, Ltr_rntk/train_size, label='RNTK', c='k', alpha=0.7, lw=2, ls='--', base=2)
plt.text(0.4, 0.7, "batch-size:{}, n_rec:{}".format(train_size, n_rec), transform=ax.transAxes)
plt.legend()

ax = plt.subplot(122)
plt.semilogx(step, Lte_bptt/test_size, label='BPTT', c='darkblue', lw=2, base=2)
plt.semilogx(step, Lte_rntk/test_size, label='RNTK', c='k', alpha=0.7, lw=2, ls='--', base=2)
plt.text(0.4, 0.7, "batch-size:{}, n_rec:{}".format(train_size, n_rec), transform=ax.transAxes)
plt.legend()
plt.show()


#%%%% save data
result_dict = {}
result_dict['X'] = X*1
result_dict['x'] = x*1
result_dict['Y'] = Y*1
result_dict['y'] = y*1
result_dict['fX_bptt'] = fX_bptt*1
result_dict['fX_rntk'] = fX_rntk*1
result_dict['fx_bptt'] = fx_bptt*1
result_dict['fx_rntk'] = fx_rntk*1
np.save("dynamics_n{}.npy".format(n_rec), result_dict)