# =============================================================================
# Code for cognitive tasks
# =============================================================================

#%% 0. Package and function
import numpy as np

def f(x):
    return np.maximum(0,x)
    
def df(x):
    return np.where(x > 0, 1, 0)

#%% 1. Model
# =============================================================================
# RNN for cognition task
# =============================================================================
class RNN:
    def __init__(self, n_in, n_rec, n_out, tau_m=100):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m

        # Initialization of parameters: pi, m, xi
        self.m_in = np.random.normal(0,1/(n_in**0.5),[n_rec,n_in])
        self.xi_in = np.zeros((n_rec,n_in))
        self.pi_in = np.zeros((n_rec,n_in))
        
        self.m_rec = np.random.normal(0,1/(n_rec**0.5),[n_rec,n_rec])
        self.xi_rec = 0.01*np.random.random((n_rec,n_rec))
        self.pi_rec = np.zeros((n_rec,n_rec))
       
        self.m_out = np.random.normal(0,1/(n_rec**0.5),[n_out,n_rec])
        self.xi_out = 0.01*np.random.random((n_out,n_rec))
        self.pi_out = np.zeros((n_out,n_rec))
       
        #important intermediate variable
        #权重w的一阶矩和二阶矩
        self.mu_in = self.m_in*(1-self.pi_in)
        self.rho_in = (1-self.pi_in)*(self.xi_in + self.m_in**2)

        self.mu_rec = self.m_rec*(1-self.pi_rec)
        self.rho_rec = (1-self.pi_rec)*(self.xi_rec + self.m_rec**2)

        self.mu_out = self.m_out*(1-self.pi_out)
        self.rho_out = (1-self.pi_out)*(self.xi_out + self.m_out**2)
                
        # rmsprop 应需要9个优化器
        self.RMSp_m_in = RMS_prop()
        self.RMSp_xi_in = RMS_prop()
        self.RMSp_pi_in = RMS_prop()
        self.RMSp_m_rec = RMS_prop()
        self.RMSp_xi_rec = RMS_prop()
        self.RMSp_pi_rec = RMS_prop()
        self.RMSp_m_out = RMS_prop()
        self.RMSp_xi_out = RMS_prop()
        self.RMSp_pi_out = RMS_prop()  
        
        #adam 需要9个优化器
        self.Adam_m_in = Adam()
        self.Adam_xi_in = Adam()
        self.Adam_pi_in = Adam()
        self.Adam_m_rec = Adam()
        self.Adam_xi_rec = Adam()
        self.Adam_pi_rec = Adam()
        self.Adam_m_out = Adam()
        self.Adam_xi_out = Adam()
        self.Adam_pi_out = Adam()  

    def uniform_pi(self):
        self.pi_rec = np.random.random((self.n_rec,self.n_rec))
        self.pi_out = np.random.random((self.n_out,self.n_rec))        
    
    def zeros_pi(self):
        self.pi_rec = np.zeros((self.n_rec,self.n_rec))
        self.pi_out = np.zeros((self.n_out,self.n_rec))    
        
    def input_mask(self):
        self.xi_in = np.zeros((self.n_rec,self.n_in))
        self.pi_in = np.zeros((self.n_rec,self.n_in))
        
        # 0-50 doen't receive visual
        self.m_in[0:50,(0,2)]=0
        self.xi_in[0:50,(0,2)]=0
        self.pi_in[0:50,(0,2)]=1
        
        # 50-100 doen't receive auditory
        self.m_in[50:100,(1,3)]=0
        self.xi_in[50:100,(1,3)]=0
        self.pi_in[50:100,(1,3)]=1        

        # 100-150 doen't receive both
        self.m_in[100:150,0:4]=0
        self.xi_in[100:150,0:4]=0
        self.pi_in[100:150,0:4]=1          
    
    def update_moment(self):
        self.mu_in = self.m_in*(1-self.pi_in)
        self.rho_in = (1-self.pi_in)*(self.xi_in + self.m_in**2)

        self.mu_rec = self.m_rec*(1-self.pi_rec)
        self.rho_rec = (1-self.pi_rec)*(self.xi_rec + self.m_rec**2)

        self.mu_out = self.m_out*(1-self.pi_out)
        self.rho_out = (1-self.pi_out)*(self.xi_out + self.m_out**2)         
     
    # variance=0, grad->0    
    def variance_fix(self,delta):
        return np.int64(delta>0)
    
    def psychometric(self,y):
        choice = 1-np.argmax(y[-1,:,:],axis=0)
        return choice
    
    def run_bptt(self, x, y_, mask, h0, eta, dt, optimizer='SGD',learning=True):
        # feedforward
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]

        gm_in, gm_rec, gm_out = 0, 0, 0  # gradients of parameters m
        gxi_in, gxi_rec, gxi_out = 0, 0, 0  # gradients of parameters xi
        gpi_in, gpi_rec, gpi_out = 0, 0, 0  # gradients of parameters pi
        
        # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, self.n_rec, b_size)) 
        h = np.zeros((t_max+1, self.n_rec, b_size))   
        
        u = np.zeros((t_max, self.n_rec, b_size))  
        g = np.zeros((t_max, self.n_rec, b_size))
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1 = np.random.normal(0,1,(t_max, self.n_rec, b_size))        

        # output 
        g_out = np.zeros((t_max,self.n_out,b_size))
        delta_out = np.zeros((t_max,self.n_out,b_size))
        epsi2 = np.random.normal(0,1,(t_max, self.n_out, b_size))                 
        y = np.zeros((t_max,self.n_out,b_size))  # RNN output
        err = np.zeros((t_max,self.n_out,b_size))  # readout error     
        
        sigma_rec = 0.15
        alpha = dt/self.tau_m
             
        # feedforward  
        self.input_mask()
        self.update_moment()   
        
        h[-1] = 1*h0
        r[-1] = f(h[-1])           
        for tt in range(-1, t_max-1, 1):
            g[tt+1] = np.dot(self.mu_rec, r[tt]) + np.dot(self.mu_in, x[tt+1])
            delta[tt+1] = 1e-30 + np.sqrt(np.dot(self.rho_rec-self.mu_rec**2, r[tt]**2) \
                                + np.dot(self.rho_in-self.mu_in**2, x[tt+1]**2))              
            u[tt+1] = g[tt+1] + delta[tt+1] * epsi1[tt+1] 
            h[tt+1] = h[tt] + alpha*(-h[tt] + u[tt+1])\
                + np.sqrt(2*alpha*sigma_rec**2)*np.random.randn(self.n_rec, b_size)
            r[tt+1] = f(h[tt+1])
            g_out[tt+1] = np.dot(self.mu_out, r[tt+1])
            delta_out[tt+1] = 1e-30 + np.sqrt(np.dot(self.rho_out-self.mu_out**2, r[tt+1]**2))       
            y[tt+1] = g_out[tt+1] + delta_out[tt+1] * epsi2[tt+1]
            
        self.r = r*1            
        if not learning:
            count = self.psychometric(y)
            return count
        
        #backpropagation
        L = 1/b_size*1/t_max*1/2*np.sum(((y - y_)*mask)**2)        
        err = 1/t_max*(y - y_)*mask     #dL/dy             
        z = np.zeros((t_max, self.n_rec, b_size))
     
        # t = T
        z[t_max-1] = df(h[t_max-1]) * np.dot((self.mu_out).T, err[t_max-1]) 
        z[t_max-1] += r[t_max-1] * df(h[t_max-1]) * np.dot((self.rho_out-self.mu_out**2).T,\
                   err[t_max-1]*epsi2[t_max-1]/delta_out[t_max-1]*self.variance_fix(delta_out[t_max-1]))      
        for tt in range(t_max-1, -1, -1):
            if tt > 0:
                z[tt-1] = z[tt]*(1 - alpha)
                z[tt-1] += alpha*df(h[tt-1])*np.dot((self.mu_rec).T, z[tt])
                z[tt-1] += alpha*df(h[tt-1])*r[tt-1]*np.dot((self.rho_rec-self.mu_rec**2).T, \
                            z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]))
                z[tt-1] += df(h[tt-1]) * np.dot((self.mu_out).T, err[tt-1]) 
                z[tt-1] += r[tt-1] * df(h[tt-1]) * np.dot((self.rho_out-self.mu_out**2).T,\
                          err[tt-1]*epsi2[tt-1]/delta_out[tt-1]*self.variance_fix(delta_out[tt-1]))                  
            
            # gradient for the theta:
            gm_rec += alpha*(1-self.pi_rec)*np.dot(z[tt], r[tt-1].T)+\
                      alpha*self.mu_rec*self.pi_rec\
                      *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]),(r[tt-1]**2).T)
            gpi_rec += -alpha*self.m_rec*np.dot(z[tt], r[tt-1].T)+\
                       -1/2*alpha*((2*self.pi_rec-1)*self.m_rec**2+self.xi_rec)\
                       *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)
            gxi_rec += 1/2*alpha*(1-self.pi_rec)\
                        *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)
            
            gm_in += alpha*(1-self.pi_in)*np.dot(z[tt], x[tt].T)+\
                     alpha*self.mu_in*self.pi_in\
                     *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]),(x[tt]**2).T)
            gpi_in += -alpha*self.m_in*np.dot(z[tt], x[tt].T)+\
                      -1/2*alpha*((2*self.pi_in-1)*self.m_in**2+self.xi_in)\
                      *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (x[tt]**2).T)
            gxi_in += 1/2*alpha*(1-self.pi_in)\
                        *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (x[tt]**2).T)
                        
            gm_out += (1-self.pi_out) * np.dot(err[tt], r[tt].T)\
                      +self.mu_out*self.pi_out\
                      *np.dot(err[tt]*epsi2[tt]/delta_out[tt]*self.variance_fix(delta_out[tt]),(r[tt]**2).T)
            gpi_out += -self.m_out*np.dot(err[tt], r[tt].T)+\
                       -1/2*((2*self.pi_out-1)*self.m_out**2+self.xi_out)\
                       *np.dot(err[tt]*epsi2[tt]/delta_out[tt]*self.variance_fix(delta_out[tt]), (r[tt]**2).T)
            gxi_out += 1/2*(1-self.pi_out)\
                        *np.dot(err[tt]*epsi2[tt]/delta_out[tt]*self.variance_fix(delta_out[tt]), (r[tt]**2).T)               


        gm_in, gm_rec, gm_out = gm_in/b_size, gm_rec/b_size, gm_out/b_size
        gxi_in, gxi_rec, gxi_out = gxi_in/b_size, gxi_rec/b_size, gxi_out/b_size
        gpi_in, gpi_rec, gpi_out = gpi_in/b_size, gpi_rec/b_size, gpi_out/b_size

        #正则化
        gamma = 1e-3
        gm_rec += gamma*self.m_rec
        gxi_rec += gamma*self.xi_rec
        gm_in += gamma*self.m_in
        gxi_in += gamma*self.xi_in
        gm_out += gamma*self.m_out
        gxi_out += gamma*self.xi_out   

       
        [eta1,eta2,eta3,eta4,eta5,eta6,eta7,eta8,eta9] = eta
        
        if optimizer=='SGD':
            self.m_in  -= eta1*gm_in
            self.xi_in -= eta2*gxi_in
            self.pi_in -= eta3*gpi_in
            self.m_rec  -= eta4*gm_rec
            self.xi_rec -= eta5*gxi_rec
            self.pi_rec -= eta6*gpi_rec
            self.m_out  -= eta7*gm_out
            self.xi_out -= eta8*gxi_out
            self.pi_out -= eta9*gpi_out
        if optimizer=='RMS':
            self.m_in = self.RMSp_m_in.New_theta(self.m_in,gm_in,eta1)
            self.xi_in = self.RMSp_xi_in.New_theta(self.xi_in,gxi_in,eta2)
            self.pi_in  = self.RMSp_pi_in.New_theta(self.pi_in,gpi_in,eta3)
            self.m_rec = self.RMSp_m_rec.New_theta(self.m_rec,gm_rec,eta4)
            self.xi_rec = self.RMSp_xi_rec.New_theta(self.xi_rec,gxi_rec,eta5)
            self.pi_rec  = self.RMSp_pi_rec.New_theta(self.pi_rec,gpi_rec,eta6)
            self.m_out = self.RMSp_m_out.New_theta(self.m_out,gm_out,eta7)
            self.xi_out = self.RMSp_xi_out.New_theta(self.xi_out,gxi_out,eta8)
            self.pi_out  = self.RMSp_pi_out.New_theta(self.pi_out,gpi_out,eta9)
        if optimizer == 'Adam':
            self.m_in = self.Adam_m_in.New_theta(self.m_in,gm_in,eta1)
            self.xi_in = self.Adam_xi_in.New_theta(self.xi_in,gxi_in,eta2)
            self.pi_in  = self.Adam_pi_in.New_theta(self.pi_in,gpi_in,eta3)
            self.m_rec = self.Adam_m_rec.New_theta(self.m_rec,gm_rec,eta4)
            self.xi_rec = self.Adam_xi_rec.New_theta(self.xi_rec,gxi_rec,eta5)
            self.pi_rec  = self.Adam_pi_rec.New_theta(self.pi_rec,gpi_rec,eta6)
            self.m_out = self.Adam_m_out.New_theta(self.m_out,gm_out,eta7)
            self.xi_out = self.Adam_xi_out.New_theta(self.xi_out,gxi_out,eta8)
            self.pi_out  = self.Adam_pi_out.New_theta(self.pi_out,gpi_out,eta9) 
        
        #normalization
        self.pi_in=np.clip(self.pi_in,0,1)
        self.xi_in=np.maximum(self.xi_in,0)
        self.pi_rec=np.clip(self.pi_rec,0,1)
        self.xi_rec=np.maximum(self.xi_rec,0)
        self.pi_out=np.clip(self.pi_out,0,1)
        self.xi_out=np.maximum(self.xi_out,0)        
        return L

# =============================================================================
# Optimizer
# =============================================================================
class RMS_prop:
    def __init__(self):
        self.lr=0.1
        self.beta=0.9
        self.epislon=1e-8
        self.s=0
        self.t=0
    
    def initial(self):
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.lr = eta
        self.t += 1
        g=gradient
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        theta -= self.lr*g/pow(self.s+self.epislon,0.5)
        return theta

class Adam:
    def __init__(self):
        self.lr=0.3
        self.beta1=0.9
        self.beta2=0.999
        self.epislon=1e-8
        self.m=0
        self.s=0
        self.t=0
    
    def initial(self):
        self.m = 0
        self.s = 0
        self.t = 0
    
    def New_theta(self,theta,gradient,eta):
        self.t += 1
        self.lr = eta
        g=gradient
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.s = self.beta2*self.s + (1-self.beta2)*(g*g)
        self.mhat = self.m/(1-self.beta1**self.t)
        self.shat = self.s/(1-self.beta2**self.t)
        theta -= self.lr*self.mhat/(pow(self.shat,0.5)+self.epislon)
        return theta
    

#%%  2. Network structure
Nin, Nrec, Nout = 5, 150, 2
tau_m = 100

#%% 3. input&output
# =============================================================================
# Input
# =============================================================================

VISUAL_P   = 0 # Positively tuned visual input
AUDITORY_P = 1 # Positively tuned auditory input
VISUAL_N   = 2 # Negatively tuned visual input
AUDITORY_N = 3 # Negatively tuned auditory input
START      = 4 # Start cue

modalities  = ['v', 'a', 'va']
freqs       = range(9, 16+1)
boundary    = 12.5
nconditions = len(modalities)*len(freqs)
pcatch      = 5/(nconditions + 1)

fmin = min(freqs)
fmax = max(freqs)

def scale_v_p(f):
    return 0.4 + 0.8*(f - fmin)/(fmax - fmin)

def scale_a_p(f):
    return 0.4 + 0.8*(f - fmin)/(fmax - fmin)

def scale_v_n(f):
    return 0.4 + 0.8*(fmax - f)/(fmax - fmin)

def scale_a_n(f):
    return 0.4 + 0.8*(fmax - f)/(fmax - fmin)


# 任务时间分段
def time_step(name, dt, catch_trial=False):
    t_catch = 2500
    if catch_trial:
        steps = {'all': list(np.arange(0,int(t_catch/dt),1))}
    else:
        if name == 'test':
            fixation = 500
        else:
            fixation = 100
        stimulus = 1000
        decision = 300
        t  = fixation + stimulus + decision

        epochs = {
            'fixation': (0, fixation),
            'stimulus': (fixation, fixation + stimulus),
            'decision': (fixation + stimulus, t)
            }
        steps = {}
        for k, (start,end) in epochs.items():
            steps[k]=list(np.arange(int(start/dt),int(end/dt),1))
    if name == 'test':
        return int(t/dt), steps
    else:
        return int(t_catch/dt), steps

def X_trial(freq, modality, t, e, dt, tau_m, catch_trial=False):
    baseline = 0.2
    alpha = dt/tau_m
    X = np.zeros((t, Nin))
    if catch_trial == False:
        if 'v' in modality:
            X[e['stimulus'],VISUAL_P] = scale_v_p(freq)
            X[e['stimulus'],VISUAL_N] = scale_v_n(freq)
        if 'a' in modality:        
            X[e['stimulus'],AUDITORY_P] = scale_a_p(freq)
            X[e['stimulus'],AUDITORY_N] = scale_a_n(freq)
        X[e['stimulus'] + e['decision'],START] = 1
    sigma_in = 0.01
    gaussian = np.random.normal(0,1,[t,Nin])
    Xin = X + baseline + 1/alpha*np.sqrt(2*alpha*sigma_in**2)*gaussian
    return Xin    

# =============================================================================
# Output
# =============================================================================
def Y_trial(freq, t, e, catch_trial = False):
    Y = np.zeros((t, Nout)) # Output matrix
    M = np.zeros_like(Y)    # Mask matrix
    # Hold values
    hi = 1.0
    lo = 0.2

    if catch_trial:
        Y[:] = lo
        M[:] = 1
    else:
        if freq > boundary:
            choice = 0
        else:
            choice = 1         
        # Fixation
        Y[e['fixation'],:] = lo
    
        # Decision
        Y[e['decision'],choice]   = hi
        Y[e['decision'],1-choice] = lo
    
        # Only care about fixation and decision periods
        M[e['fixation']+e['decision'],:] = 1
    return Y,M

    
#%% 4. Train and Test
def mini_batch(name, t, e, dt, tau_m):
    b_size = 24
    if name == 'train':
        catch_trial_size = 6
        b_size += catch_trial_size
    X_mb = np.zeros([t,Nin,b_size])
    Y_mb = np.zeros([t,Nout,b_size])
    M_mb = np.zeros([t,Nout,b_size])
    catch_trial = False
    for i,modality in enumerate(modalities):
        for j,freq in enumerate(freqs):
            X_mb[:,:,i*8+j] = 1*X_trial(freq, modality, t, e, dt, tau_m, catch_trial)
            Y, M = Y_trial(freq, t, e, catch_trial)
            Y_mb[:,:,i*8+j] = 1*Y
            M_mb[:,:,i*8+j] = 1*M
    if name == 'train':
        catch_trial = True
        for mb in range(24,24+catch_trial_size):
            X_mb[:,:,mb] = 1*X_trial(None, None, t, e, dt, tau_m, catch_trial)
            Y, M = Y_trial(None, t, e, catch_trial)
            Y_mb[:,:,mb] = 1*Y
            M_mb[:,:,mb] = 1*M
    return X_mb, Y_mb, M_mb


def test_batch(net, batch, dt, tau_m):    
    b_size = 24    
    t,e = time_step('test', dt)
    count_freq = 0
    for i in range(batch):
        X_test, Y_test, M_test = mini_batch('test', t, e, dt, tau_m) 
        h_test = 0*np.ones([Nrec, b_size])
        count = net.run_bptt(X_test, Y_test, M_test, h_test, None, dt, learning=False)
        count_freq += count
        print('\rTest Process: {:.2f}%'.format((i+1)/batch*100),end='')
    count_freq = count_freq/batch 
    P_matrix = count_freq.reshape(len(modalities),len(freqs))
    return P_matrix


def train_batch(net, learn_rate, ep, dt, tau_m):
    t,e = time_step('train', dt) 
    X,Y,M = mini_batch('train', t, e, dt, tau_m) 
    b_size = X.shape[2]
    h_init = 0*np.ones([Nrec,b_size])
    L = net.run_bptt(X, Y, M, h_init, 
            learn_rate, dt, optimizer='RMS', learning=True)  
    print('\rEpoch:{}; Train MSE = {:.4f};'.format(ep,L),end='')
    return L 

#%% 5. save data
paras = ['pi','m','xi']
layers = ['in','rec','out']

def record(data, para, layer):
    data_list = open("para_{}_{}.txt".format(para,layer),"a")
    np.savetxt(data_list,data,newline='\n')
    data_list.close()

def record_net(net):
    record(net.m_in.reshape(1,-1), 'm', 'in')
    record(net.xi_in.reshape(1,-1), 'xi', 'in')
    record(net.pi_in.reshape(1,-1), 'pi', 'in')
    record(net.m_rec.reshape(1,-1), 'm', 'rec')
    record(net.xi_rec.reshape(1,-1), 'xi', 'rec')    
    record(net.pi_rec.reshape(1,-1), 'pi', 'rec')    
    record(net.m_out.reshape(1,-1), 'm', 'out')
    record(net.xi_out.reshape(1,-1), 'xi', 'out')
    record(net.pi_out.reshape(1,-1), 'pi', 'out')   

def create_file(para, layer):
    data_list = open("para_{}_{}.txt".format(para,layer), "w")
    data_list.close()

def create_netfile():
    create_file('m', 'in')
    create_file('xi', 'in')
    create_file('pi', 'in')
    create_file('m', 'rec')
    create_file('xi', 'rec')    
    create_file('pi', 'rec')    
    create_file('m', 'out')
    create_file('xi', 'out')
    create_file('pi', 'out') 
    
#%% 6. main
def learn_rate_epoch(epoch):
    lr_m, lr_xi = 1e-3, 1e-3
    flag = (epoch+1)//100
    lr_pi_rec = 1*0.5**flag
    lr_pi_out = 1e-1*0.5**flag
    learn_rate = np.array([lr_m,0,0,\
                           lr_m,lr_xi,lr_pi_rec,\
                           lr_m,lr_xi,lr_pi_out])
    return learn_rate

net1 = RNN(Nin, Nrec, Nout, tau_m)
epochs = 500

import time 
time1 = time.time()
dt_train = 20
create_netfile()
record_net(net1)
for epoch in range(epochs): 
    lr = learn_rate_epoch(epoch)
    L = train_batch(net1, lr, epoch, dt_train, tau_m)
    record_net(net1)
    if epoch>=100:
        if L<=0.0020:
            break      
time2 = time.time()  
print('\ntime cost = {:.2f}'.format(time2-time1))
