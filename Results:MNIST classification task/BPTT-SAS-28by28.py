#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np



#%% 定义函数
def f(x):
    return np.maximum(0,x)

def df(x):
    return np.where(x > 0, 1, 0)
'''
def gradient_clip(gra,bound):
    l2=(np.linalg.norm(gra,ord=2))
    if l2>=bound:
        return np.array((bound/l2)*gra)
    else:
        return np.array(gra)
'''
def grad_clip(gra,bound):
    l2=(np.linalg.norm(gra,ord=2))
    if l2>=bound:
        return np.array((bound/l2)*gra)
    else:
        return np.array(gra)

def decay(epoch,ddl,lr0,step):
    lr = (0.5**(epoch//step))*lr0
    lr[lr<=ddl] = ddl
    return lr
def sampling(m,p,v):
    import random
    m=np.array(m)
    p=np.array(p)
    v=np.array(v)
    b=np.ones((p.shape[0],p.shape[1]))
    ran=np.array(np.random.random((p.shape[0],p.shape[1]) ))
    for i in range(0,p.shape[0]):
        for j in range(0,p.shape[1]):
            if v[i][j]==0:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*m[i][j]
            else:
                b[i][j]=(turn_2_zero((ran[i][j]-p[i][j])))*(np.random.normal(m[i][j],np.sqrt(v[i][j])))
    return b
def turn1(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat))+np.eye(N))
def turn0(mat):
    N=mat.shape[0]
    return(mat-np.diag(np.diag(mat)))
def softmax(y):
    y = y - np.array(y.max(axis=0),ndmin=2)
    exp_y = np.exp(y) 
    sumofexp = np.array(exp_y.sum(axis=0),ndmin=2)
    softmax = exp_y/sumofexp
    return softmax


# In[22]:


#%% 定义RNN类
class RNN:
    def __init__(self, n_in, n_rec, n_out, tau_m=10):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out
        self.tau_m = tau_m

        # Initialization of parameters: pi, m, xi
        # 用于刻画权重w的分布
        # 后一层在第一维度
        self.m_in =  np.array((np.random.normal(0,1,(n_rec,n_in)))/n_rec**0.5*n_in**0.5)
        self.xi_in = 0.1*np.random.random((n_rec,n_in))
        self.pi_in = np.zeros((n_rec,n_in))
        
        self.m_rec = np.identity(n_rec)
        self.xi_rec = 0.1*np.random.random((n_rec,n_rec))
        self.pi_rec = (np.zeros((n_rec,n_rec)))
        
        self.m_out =  np.array((np.random.normal(0,1,(n_out,n_rec)))/n_rec**0.5*n_in**0.5)
        self.xi_out = 0.1*np.random.random((n_out,n_rec))
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
        self.Adam_pi_in = Adam2()
        self.Adam_m_rec = Adam()
        self.Adam_xi_rec = Adam()
        self.Adam_pi_rec = Adam2()
        self.Adam_m_out = Adam()
        self.Adam_xi_out = Adam()
        self.Adam_pi_out = Adam2()  
                   
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
    def sample_from_p(self,pi,m,xi):
        rand = np.random.random(np.shape(pi))
        w = np.random.normal(m,np.sqrt(xi))
        w_0 = self.turn_2_zero(rand-pi)
        return w*w_0
    def turn_2_zero(self,x):
        return np.int64(x>0)
    def sample_w(self):
        self.w_in = self.sample_from_p(self.pi_in, self.m_in, self.xi_in)
        self.w_rec = self.sample_from_p(self.pi_rec, self.m_rec, self.xi_rec)
        self.w_out = self.sample_from_p(self.pi_out, self.m_out, self.xi_out)
    
    def sample_forward(self,x,y_,h0):
        # feedforward
        self.sample_w()
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]
        r = np.zeros((t_max+1, self.n_rec, b_size)) 
        h = np.zeros((t_max+1, self.n_rec, b_size))
        u = np.zeros((t_max, self.n_rec, b_size))  
        y = np.zeros((self.n_out,b_size)) 
        h[-1] = 1*h0
        r[-1] = f(h[-1])
        for tt in range(-1,t_max-1,1):
            self.sample_w()
            u[tt+1] = np.dot(self.w_rec, r[tt])+ np.dot(self.w_in, x[tt+1])
            h[tt+1] = h[tt] + (-h[tt] + u[tt+1])/self.tau_m
            r[tt+1] = f(h[tt+1])
        y = np.dot(self.w_out, r[t_max-1])
        y=softmax(y)
    
            
        
        
        return self.accuracy(y,y_)
    
    
    
    def accuracy(self,y,y_):
        targets = y_.argmax(axis=0)
        predicts = y.argmax(axis=0)
        return np.sum(targets==predicts)/np.size(targets)
    
    def run_bptt(self, x, y_, h0, eta, optimizer='SGD',learning=True):
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]

        gm_in, gm_rec, gm_out = 0, 0, 0  # gradients of parameters m
        gxi_in, gxi_rec, gxi_out = 0, 0, 0  # gradients of parameters xi
        gpi_in, gpi_rec, gpi_out = 0, 0, 0  # gradients of parameters pi
    
        # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, self.n_rec, b_size)) 
        h = np.zeros((t_max+1, self.n_rec, b_size))
        # 由于存在初始值h0，用h[-1]（即h[t_max]）存放h，r同理
        # h[t_max-1]仍标记演化的最后一时间步, r同理
    
        u = np.zeros((t_max, self.n_rec, b_size))  
        g = np.zeros((t_max, self.n_rec, b_size))
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1=np.random.normal(0,1,(t_max,self.n_rec, b_size))
        epsi2=np.random.normal(0,1,(self.n_out, b_size))        
        

        # output 
        g_out = np.zeros((self.n_out,b_size))
        delta_out = np.zeros((self.n_out,b_size))         
        y = np.zeros((self.n_out,b_size))  # RNN output
        err = np.zeros((self.n_out,b_size))  # readout error     
    
        self.update_moment()
    
        # feedforward    
        h[-1] = 1*h0
        r[-1] = f(h[-1])

        for tt in range(-1,t_max-1,1):
            g[tt+1] = np.dot(self.mu_rec, r[tt])+ np.dot(self.mu_in, x[tt+1])
            delta[tt+1] = np.sqrt(np.dot(self.rho_rec-self.mu_rec**2, r[tt]**2)                             + np.dot(self.rho_in-self.mu_in**2, x[tt+1]**2))              
            u[tt+1] = g[tt+1] + delta[tt+1] * epsi1[tt+1]
            h[tt+1] = h[tt] + (-h[tt] + u[tt+1])/self.tau_m
            r[tt+1] = f(h[tt+1])
        g_out = np.dot(self.mu_out, r[t_max-1])
        delta_out = np.sqrt(np.dot(self.rho_out-self.mu_out**2, r[t_max-1]**2))       
        y = g_out + delta_out * epsi2
    
        if not learning:
            accuracy = self.accuracy(y, y_)
            return accuracy
    
    #softmax层
        softmax1=softmax(y)
        CE_k = (-y_*np.log(softmax1+pow(10,-5))).sum(axis=0)
        CE = np.sum(CE_k)/np.size(CE_k)

    #backpropagation
        err = softmax1 - y_     #dL/dy             
        z = np.zeros((t_max, self.n_rec, b_size))
        delta1 = pow(10,-30)*np.ones_like((delta))+delta
        delta_out1 = pow(10,-30)*np.ones_like((delta_out))+delta_out
    # t = T
        z[t_max-1] =  df(h[t_max-1]) * np.dot((self.mu_out).T, err)  
        z[t_max-1] += r[t_max-1]* df(h[t_max-1]) * np.dot((self.rho_out-self.mu_out**2).T,                     err*epsi2/delta_out1*self.variance_fix(delta_out))   
        for tt in range(t_max-1, -1, -1):
        # gradient for the theta:
            gm_rec += 1/(self.tau_m)*(1-self.pi_rec)*np.dot(z[tt], r[tt-1].T)+                  1/(self.tau_m)*self.mu_rec*self.pi_rec                  *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]),(r[tt-1]**2).T)
            gpi_rec += -1/(self.tau_m)*self.m_rec*np.dot(z[tt], r[tt-1].T)+                   -1/(2*self.tau_m)*((2*self.pi_rec-1)*self.m_rec**2+self.xi_rec)                   *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)
            gxi_rec += 1/(2*self.tau_m)*(1-self.pi_rec)                    *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)
        
            gm_in += 1/(self.tau_m)*(1-self.pi_in)*np.dot(z[tt], x[tt].T)+                 1/(self.tau_m)*self.mu_in*self.pi_in                 *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]),(x[tt]**2).T)
            gpi_in += -1/(self.tau_m)*self.m_in*np.dot(z[tt], x[tt].T)+                  -1/(2*self.tau_m)*((2*self.pi_in-1)*self.m_in**2+self.xi_in)                  *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]), (x[tt]**2).T)
            gxi_in += 1/(2*self.tau_m)*(1-self.pi_in)                    *np.dot(z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]), (x[tt]**2).T)
        
            if tt > 0:
                z[tt-1] = z[tt]*(1 - 1/self.tau_m)
                z[tt-1] += 1/self.tau_m*df(h[tt-1])*np.dot((self.mu_rec).T, z[tt])
                z[tt-1] += 1/self.tau_m*df(h[tt-1])*r[tt-1]*np.dot((self.rho_rec-self.mu_rec**2).T,                        z[tt]*epsi1[tt]/delta1[tt]*self.variance_fix(delta[tt]))
    

        # 计算Wout相关的梯度
        gm_out += (1-self.pi_out)*np.dot(err, r[t_max-1].T)+              self.mu_out*self.pi_out              *np.dot(err*epsi2/delta_out1*self.variance_fix(delta_out),(r[t_max-1]**2).T)
        gpi_out += -self.m_out*np.dot(err, r[t_max-1].T)+               -1/2*((2*self.pi_out-1)*self.m_out**2+self.xi_out)               *np.dot(err*epsi2/delta_out1*self.variance_fix(delta_out), (r[t_max-1]**2).T)
        gxi_out += 1/2*(1-self.pi_out)                *np.dot(err*epsi2/delta_out1*self.variance_fix(delta_out), (r[t_max-1]**2).T)      
    
    #梯度裁剪
        gm_in, gm_rec, gm_out = grad_clip(gm_in/b_size,10), grad_clip(gm_rec/b_size,10), grad_clip(gm_out/b_size,10)
        gxi_in, gxi_rec, gxi_out = grad_clip(gxi_in/b_size,10), grad_clip(gxi_rec/b_size,10), grad_clip(gxi_out/b_size,10)
        gpi_in, gpi_rec, gpi_out =grad_clip(gpi_in/b_size,10), grad_clip(gpi_rec/b_size,10), grad_clip(gpi_out/b_size,10)
    
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
        self.pi_rec=(np.clip(self.pi_rec,0,1))
        self.xi_rec=(np.maximum(self.xi_rec,0))
        self.pi_out=np.clip(self.pi_out,0,1)
        self.xi_out=np.maximum(self.xi_out,0)          
        return CE

    
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
        self.decay=1e-4
        self.s = self.beta*self.s + (1-self.beta)*(g*g)
        theta -= self.lr*((g/pow(self.s+self.epislon,0.5))+self.decay*theta)
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
        self.decay=1e-4
        g=gradient
        self.m = self.beta1*self.m + (1-self.beta1)*g
        self.s = self.beta2*self.s + (1-self.beta2)*(g*g)
        self.mhat = self.m/(1-self.beta1**self.t)
        self.shat = self.s/(1-self.beta2**self.t)
        theta -= self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon))+self.decay*theta)
        return theta
#No decay for pi
class Adam2:
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
        theta -= self.lr*((self.mhat/(pow(self.shat,0.5)+self.epislon)))
        return theta





#%% 训练和测试函数
def data_shuffle(data,label):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)
    return data,label
    
def train(net, b_size, batches, train_data, train_label, learn_rate, optimizer):
    #设置批次
    data,targets = data_shuffle(train_data,train_label)
    #计算均值
    CEs = 0   
    h_init = 0.1*np.ones([n_rec,b_size])    
    for ii in range(batches):
        x = (data[ii*b_size:(ii+1)*b_size]).T.reshape(28,28,b_size)
        y_ = (targets[ii*b_size:(ii+1)*b_size]).T
        CE = net1.run_bptt(x, y_, h_init, eta=learn_rate, optimizer=optimizer)     
        CEs += CE  
        print('\rTraning process: {:.2f}%.'.format(100*(ii+1)/batches),end='')
    return CEs/batches

def test(net, test_data, test_label,sampling=False):  
    #计算总数
    accuracies = 0  
    #设置批次
    batches = 10
    b_size = 1000    
    data = test_data
    targets = test_label
    h_test = 0.1*np.ones([n_rec,b_size])
    for ii in range(batches):      
        x_test = (data[ii*b_size:(ii+1)*b_size].T).reshape(28,28,b_size)      
        y_test = (targets[ii*b_size:(ii+1)*b_size]).T
        if sampling:
            accuracy = net1.sample_forward(x_test,y_test,h_test)
        else:
            accuracy = net1.run_bptt(x_test,y_test,h_test,eta=learn_rate,learning=False)
        accuracies += accuracy
    return accuracies/batches 


# In[23]:


import loadm
mnist=np.array(loadm.load_mnist(one_hot=True))
train_data = mnist[0][0][0:60000]
train_label = mnist[0][1][0:60000]
test_data = mnist[1][0][:10000]
test_label = mnist[1][1][:10000]
print(np.shape(train_data))
print(np.shape(train_label))


#%%
#设置参数
batches = 200
b_size = 128
learn_rate =  np.array([pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3),pow(10,-3)])
n_in, n_rec, n_out = 28, 100, 10
optimizer = 'Adam'
net1 = RNN(n_in, n_rec, n_out)


# In[24]:


import time
start = time.time()
print('Training begin.')
CEs = []
accuracies = []
Total_epoch = 200
m_in_all=[]
m_out_all=[]
m_rec_all=[]
pi_in_all=[]
pi_out_all=[]
pi_rec_all=[]
sig_in_all=[]
sig_out_all=[]
sig_rec_all=[]        
        
for epoch in range(Total_epoch):
    print('Epoch {}'.format(epoch))
    m_in_all.append(net1.m_in*1)
    m_out_all.append(net1.m_out*1)
    m_rec_all.append(net1.m_rec*1)
            
    pi_in_all.append(net1.pi_in*1)
    pi_out_all.append(net1.pi_out*1)
    pi_rec_all.append(net1.pi_rec*1)
            
    sig_in_all.append(net1.xi_in*1)
    sig_out_all.append(net1.xi_out*1)
    sig_rec_all.append(net1.xi_rec*1)    
    np.save('RNN-SAS-28/1/RNN-SAS-m_in_all',m_in_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-pi_in_all',pi_in_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-sigma_in_all',sig_in_all, allow_pickle=True)
    
    np.save('RNN-SAS-28/1/RNN-SAS-m_out_all',m_out_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-pi_out_all',pi_out_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-sigma_out_all',sig_out_all, allow_pickle=True)
            
    np.save('RNN-SAS-28/1/RNN-SAS-m_rec_all',m_rec_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-pi_rec_all',pi_rec_all, allow_pickle=True)
    np.save('RNN-SAS-28/1/RNN-SAS-sigma_rec_all',sig_rec_all, allow_pickle=True)





    lr = decay(epoch+1, 1e-8, learn_rate, 40)
    CE = train(net1, b_size, batches, train_data, train_label, lr, optimizer)    
    CEs.append(CE)
    accuracy = test(net1, test_data, test_label,sampling=False)
    accuracies.append(accuracy)
    accuracy2=test(net1, test_data, test_label,sampling=True)
    print(' CE = {:.4f}; Accuracy = {:.2f}%.'.format(CE,accuracy*100))
    print(' CE = {:.4f}; Sampling Accuracy = {:.2f}%.'.format(CE,accuracy2*100))
    np.save('RNN-SAS-28/1/accuracy',accuracies, allow_pickle=True)    
end = time.time()


# In[ ]:




