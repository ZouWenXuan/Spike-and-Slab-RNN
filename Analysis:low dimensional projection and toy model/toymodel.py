#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt


def divi_(lr0,global_step,decay_step):
    return lr0*(0.5**((int(global_step/decay_step))))

#%% 定义函数
def f(x):
    return np.tanh(x)

def df(x):
    return (1-np.tanh(x)**2)
class RNN:
    def __init__(self, n_rec, a,tau_m=1/(0.9)):
        self.n_rec = n_rec
        self.tau_m = tau_m
        self.a = a
        self.m_rec = np.sqrt(1/self.n_rec)*np.random.normal(0,1,(1,n_rec))
        self.xi_rec = 0.1*np.random.random((1,n_rec))
        self.pi_rec = 0.1*np.random.random((1,n_rec))
        self.mu = (1-self.pi_rec)*self.m_rec
        self.rho = (1-self.pi_rec)*(self.m_rec**2+self.xi_rec)
    def update(self):
        self.mu = (1-self.pi_rec)*self.m_rec
        self.rho = (1-self.pi_rec)*(self.m_rec**2+self.xi_rec)
    def variance_fix(self,delta):
        delt = delta*1
        delt[delta>0] = 1
        delt[delta<0] = 0
        return delt
    def sample_from_p(self,pi,m,xi):
        rand = np.random.random(np.shape(pi))
        w = np.random.normal(m,np.sqrt(xi))
        w_0 = self.turn_2_zero(rand-pi)
        return w*w_0
    def turn_2_zero(self,x):
        return np.int64(x>0)
    def sample_w(self):
        self.w_rec = self.sample_from_p(self.pi_rec, self.m_rec, self.xi_rec)
#     def evaluate(self,x1,y1):
#         ##x输入为 t_max*n_rec
#         ## y 输入为 t_max*1
#         x = np.array(x1)
#         y = np.array(y1) 
#         t_max = np.shape(x)[0]  
#         h0 = 0.5*np.ones([1])
#          # input recurrent (feedforward plus recurrent)
#         r = np.zeros((t_max+1, 1)) 
#         h = np.zeros((t_max+1, 1))
#         h[-1] = 1*h0
#         r[-1] = f(h[-1])
#         alpha = 1/self.tau_m
#         CE = 0
#         self.update()
#         self.sample_w()
#         for tt in range(-1,t_max-1,1):
#             self.update()
#             h[tt+1] =(1-alpha)*h[tt]+alpha*(np.sum(self.w_rec*x[tt+1]))
#             r[tt+1] = f(h[tt+1])
#             CE = CE + np.sum((r[tt+1] - y[tt+1])**2)/2.0
#         return CE,r
    def evaluate(self,x1,y1):
        ##x输入为 t_max*n_rec
        ## y 输入为 t_max*1
        x = np.array(x1)
        y = np.array(y1) 
        t_max = np.shape(x)[0]  
        h0 = 0.0*np.ones([1])
         # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, 1)) 
        h = np.zeros((t_max+1, 1))
        G = np.zeros((t_max+1, 1))
        delta = np.zeros((t_max+1, 1))
        epsilon = 0*np.random.normal(0,1,(t_max,1))
        h[-1] = 1*h0
        r[-1] = f(h[-1])
        alpha = 1/self.tau_m
        CE = 0
        a=self.a
        for tt in range(-1,t_max-1,1):
            self.update()
            G[tt+1] = 1/(np.sqrt(a))*np.sum(self.mu*x[tt+1])
            delta[tt+1] = 1/(np.sqrt(a))*np.sqrt(np.sum((self.rho-self.mu**2)*(x[tt+1]**2)))  
            h[tt+1] =(1-alpha)*h[tt]+alpha*(G[tt+1]+epsilon[tt+1]*(delta[tt+1]))
            r[tt+1] = f(h[tt+1])
            CE = CE + np.sum((r[tt+1] - y[tt+1])**2)/2.0
        return CE,r
    def single_up(self,x1,y1,lr1,lr2,lr3):
        x = np.array(x1)
        y = np.array(y1) 
        t_max = np.shape(x)[0]  
        h0 = 0.0*np.ones([1])
         # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, 1)) 
        h = np.zeros((t_max+1, 1))
        G = np.zeros((t_max+1, 1))
        delta = np.zeros((t_max+1, 1))
        delta1 = np.zeros((t_max+1, 1))
        epsilon = np.random.normal(0,1,(t_max,1))
        h[-1] = 1*h0
        r[-1] = f(h[-1])
        alpha = 1/self.tau_m
        CE = 0
        a = self.a
        forward_mean = []
        forward_variance=[]
        gra_mean=[]
        gra_variance=[]
        for tt in range(-1,t_max-1,1):
            self.update()
            G[tt+1] = 1/(np.sqrt(a))*np.sum(self.mu*x[tt+1])
            delta[tt+1] =  1/(np.sqrt(a))*np.sqrt(np.sum((self.rho-self.mu**2)*(x[tt+1]**2))) 
            delta1[tt+1] = pow(10,-30)*np.ones_like((delta[tt+1]))+delta[tt+1]
            h[tt+1] =(1-alpha)*h[tt]+alpha*(G[tt+1]+epsilon[tt+1]*(delta[tt+1]))
            r[tt+1] = f(h[tt+1])
            forward_mean.append(np.abs(G[tt+1])*1)
            forward_variance.append((np.abs(delta[tt+1]))*1)
            self.update()
            nabla_m = (r[tt+1]-y[tt+1])*df(h[tt+1])*alpha*((1-self.pi_rec)*x[tt+1]*(1/(np.sqrt(a)))+(1/a)*epsilon[tt+1]*self.mu*self.pi_rec*(x[tt+1]**2)/delta1[tt+1]*self.variance_fix(delta[tt+1]))
            self.m_rec = self.m_rec-lr1*nabla_m
            gra_mean.append(np.average(np.abs((1-self.pi_rec)*x[tt+1]*(1/(np.sqrt(a)))*1))*1)
            gra_variance.append(np.average(np.abs((1/a)*self.mu*self.pi_rec*(x[tt+1]**2)/delta1[tt+1]*self.variance_fix(delta[tt+1])*1)*1))
            nabla_pi =(r[tt+1]-y[tt+1])*df(h[tt+1])*alpha*(-self.m_rec*x[tt+1]*(1/(np.sqrt(a)))+(1/a)*epsilon[tt+1]*(((self.m_rec**2)*(1-2*self.pi_rec)-self.xi_rec)*(x[tt+1]**2)/(2*delta1[tt+1]))*self.variance_fix(delta[tt+1]))
            self.pi_rec =np.clip((self.pi_rec - lr2*nabla_pi),0,1)
            nabla_xi = (r[tt+1]-y[tt+1])*df(h[tt+1])*alpha*((1/a)*(epsilon[tt+1]*(1-self.pi_rec)*(x[tt+1]**2)/(2*delta1[tt+1]))*self.variance_fix(delta[tt+1]))
            self.xi_rec = np.maximum((self.xi_rec- lr3*nabla_xi),0)
        return np.average(forward_mean),np.average(forward_variance),np.average(gra_mean),np.average(gra_variance)
                
    def SGD(self,x1,y1,epoch,lr10,lr20,lr30):
        loss = []
        x = np.array(x1)
        y = np.array(y1)
        ffm=[]
        ffv=[]
        ggm=[]
        ggv=[]
        for i in range(epoch):
            self.update()
            lr1 = divi_(lr10,i,100)
            lr2 = divi_(lr20,i,100)
            lr3 = divi_(lr30,i,100)
            loss.append(self.evaluate(x,y)[0])
            fm,fv,gm,gv = self.single_up(x,y,lr1,lr2,lr3)
            ffm.append(fm*1)
            ffv.append(fv*1)
            ggm.append(gm*1)
            ggv.append(gv*1)
        return loss,ffm,ffv,ggm,ggv


# In[4]:


MSE = []
for i in range(10):
    n_rec=10000
    seq_length = 20
    data_time_steps = np.linspace(2,10,seq_length+1)
    label=np.sin(data_time_steps)
    label = label.reshape(seq_length+1,1)#21*1
    data = label*1
    net1 = RNN(n_rec,n_rec)
    acc1,ffm,ffv,ggm,ggv = net1.SGD(data,label,600,0.005,0.01,0.01)
    print(acc1[-1])
    MSE.append(acc1)







