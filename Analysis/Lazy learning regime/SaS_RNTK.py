# =============================================================================
# Recurrent Neural Tangent Kernel for SaS RNN
# =============================================================================

import numpy as np
# import matplotlib.pyplot as plt


#%% SaS RNN
class RNN:
    def __init__(self, n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out

        # activation function
        self.f = lambda x: np.maximum(0, x)
        self.df = lambda x: np.where(x>0, 1, 0)

        # RNTK parameter
        self.sigma_in = sigma_in
        self.sigma_rec = sigma_rec
        self.sigma_out = sigma_out
        self.sigma_h0 = sigma_h0

        # trainable parameter
        # input
        self.m_in_tr = np.random.normal(0,1,[self.n_rec,self.n_in])
        self.xi_in_tr = np.zeros((self.n_rec,self.n_in))
        self.pi_in_tr = np.zeros((self.n_rec,self.n_in))
        # recurrent
        self.m_rec_tr = np.random.normal(0,1,[self.n_rec,self.n_rec])
        self.xi_rec_tr = np.random.random((self.n_rec,self.n_rec))
        self.pi_rec_tr = np.zeros((self.n_rec,self.n_rec))
       # output
        self.m_out_tr = np.random.normal(0,1,[self.n_out,self.n_rec])
        self.xi_out_tr = np.zeros((self.n_out,self.n_rec))
        self.pi_out_tr = np.zeros((self.n_out,self.n_rec))

    
    def rescale(self):
        # input
        self.m_in = self.sigma_in/(self.n_in**0.5)*self.m_in_tr
        self.xi_in = 1*self.xi_in_tr
        self.pi_in = 1*self.pi_in_tr
        # recurrent
        self.m_rec = self.sigma_rec/(self.n_rec**0.5)*self.m_rec_tr
        self.xi_rec = 1/self.n_rec*self.xi_rec_tr
        self.pi_rec = 1*self.pi_rec_tr
        # output
        self.m_out = self.sigma_out/(self.n_rec**0.5)*self.m_out_tr
        self.xi_out = 1*self.xi_out_tr
        self.pi_out = 1*self.pi_out_tr
    
    
    def normalization(self):
        self.pi_in=np.clip(self.pi_in,0,1)
        self.xi_in=np.maximum(self.xi_in,0)
        self.pi_rec=np.clip(self.pi_rec,0,1)
        self.xi_rec=np.maximum(self.xi_rec,0)
        self.pi_out=np.clip(self.pi_out,0,1)
        self.xi_out=np.maximum(self.xi_out,0)  


    def update_moment(self):
        self.mu_in = self.m_in*(1-self.pi_in)
        self.rho_in = (1-self.pi_in)*(self.xi_in + self.m_in**2)

        self.mu_rec = self.m_rec*(1-self.pi_rec)
        self.rho_rec = (1-self.pi_rec)*(self.xi_rec + self.m_rec**2)

        self.mu_out = self.m_out*(1-self.pi_out)
        self.rho_out = (1-self.pi_out)*(self.xi_out + self.m_out**2)  
    
    
    def SaS_output(self):
        self.xi_out = 1/self.n_rec*self.xi_out_tr
        self.pi_out = 1*self.pi_out_tr
    
    
    # variance=0, grad->0    
    def variance_fix(self, delta):
        return np.where(delta>1e-29, 1, 0)
    
    
    def expand_bsize(self, gr, gc):
        # gr = (Nr, b_size) -> (b_size, Nr, 1)
        # gc = (b_size, Nc) -> (b_size, 1, Nc)
        gr = np.expand_dims(gr.T, 2)
        gc = np.expand_dims(gc, 1)
        g = gr*gc
        return g
    
    
    def gradient_bptt(self, x, alpha, check=None, SaS_output=False):
        # feedforward
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]

        gm_in, gm_rec, gm_out = 0, 0, 0
        gpi_rec, gxi_rec = 0, 0
        if SaS_output:
            gpi_out, gxi_out = 0, 0
            
        # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, self.n_rec, b_size)) 
        h = np.zeros((t_max+1, self.n_rec, b_size))    
        
        u = np.zeros((t_max, self.n_rec, b_size))  
        g = np.zeros((t_max, self.n_rec, b_size))
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1 = np.random.normal(0,1,(t_max, self.n_rec, b_size))        

        # single output 
        g_out = np.zeros((self.n_out, b_size))
        delta_out = np.zeros((self.n_out, b_size))
        epsi2 = np.random.normal(0,1,(self.n_out, b_size))                 
        y = np.zeros((self.n_out, b_size))  # RNN output
        err = np.zeros((self.n_out, b_size))  # readout error     
        
        # feedforward
        self.rescale()
        self.normalization()
        self.update_moment()
        h0 = self.sigma_h0 * np.random.randn(self.n_rec, b_size)
        h[-1] = 1*h0
        r[-1] = self.f(h[-1])           
        for tt in range(-1, t_max-1, 1):
            g[tt+1] = np.dot(self.mu_rec, r[tt]) + np.dot(self.mu_in, x[tt+1])
            delta[tt+1] = 1e-30 + np.sqrt(np.dot(self.rho_rec-self.mu_rec**2, r[tt]**2) \
                                + np.dot(self.rho_in-self.mu_in**2, x[tt+1]**2))              
            u[tt+1] = g[tt+1] + delta[tt+1] * epsi1[tt+1] 
            h[tt+1] = h[tt] + alpha*(-h[tt] + u[tt+1])
            r[tt+1] = self.f(h[tt+1])
        if check == "Sigma":
            return u,h
        
        g_out = np.dot(self.mu_out, r[t_max-1])
        delta_out = 1e-30 + np.sqrt(np.dot(self.rho_out-self.mu_out**2, r[t_max-1]**2))       
        y = g_out + delta_out * epsi2

        # backpropagation   
        err = y/y                 
        z = np.zeros((t_max, self.n_rec, b_size))
        v = np.zeros((t_max-1, self.n_rec, b_size))
        
        # t = T
        z[t_max-1] = self.df(h[t_max-1]) * np.dot((self.mu_out).T, err) 
        z[t_max-1] += r[t_max-1] * self.df(h[t_max-1]) * np.dot((self.rho_out-self.mu_out**2).T,\
                   err*epsi2/delta_out*self.variance_fix(delta_out))      
        for tt in range(t_max-1, -1, -1):
            if tt > 0:
                z[tt-1] = z[tt]*(1 - alpha)
                v[tt-1] = self.df(h[tt-1])*( np.dot((self.mu_rec).T, z[tt]) \
                            + r[tt-1]*np.dot((self.rho_rec-self.mu_rec**2).T, \
                              z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt])) )  
                z[tt-1] += alpha*v[tt-1]
             
            # gradient for the theta:
            gm_rec += alpha*(1-self.pi_rec)*self.expand_bsize(z[tt], r[tt-1].T)+\
                      alpha*self.mu_rec*self.pi_rec\
                      *self.expand_bsize(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]),(r[tt-1]**2).T)
            gpi_rec += -alpha*self.m_rec*self.expand_bsize(z[tt], r[tt-1].T)+\
                        -1/2*alpha*((2*self.pi_rec-1)*self.m_rec**2+self.xi_rec)\
                        *self.expand_bsize(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)
            gxi_rec += 1/2*alpha*(1-self.pi_rec)\
                        *self.expand_bsize(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]), (r[tt-1]**2).T)           
            gm_in += alpha*(1-self.pi_in)*self.expand_bsize(z[tt], x[tt].T)+\
                     alpha*self.mu_in*self.pi_in\
                     *self.expand_bsize(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]),(x[tt]**2).T)

        # gradient of output layer               
        gm_out += (1-self.pi_out) * self.expand_bsize(err, r[t_max-1].T)\
                  +self.mu_out*self.pi_out\
                  *self.expand_bsize(err*epsi2/delta_out*self.variance_fix(delta_out),(r[t_max-1]**2).T)
        if SaS_output:
            gpi_out += -self.m_out * self.expand_bsize(err, r[t_max-1].T)\
                       -1/2*((2*self.pi_out-1)*self.m_out**2+self.xi_out)\
                       *self.expand_bsize(err*epsi2/delta_out*self.variance_fix(delta_out),(r[t_max-1]**2).T)
            gxi_out += 1/2*(1-self.pi_out)\
                       *self.expand_bsize(err*epsi2/delta_out*self.variance_fix(delta_out),(r[t_max-1]**2).T)
        if check == "Pi":
            return z, v
        
        # return results
        self.gm_in, self.gm_rec, self.gm_out = gm_in*1, gm_rec*1, gm_out*1
        self.gpi_rec, self.gxi_rec = gpi_rec*1, gxi_rec*1
        if SaS_output:
            self.gpi_out, self.gxi_out = gpi_out*1, gxi_out*1
     

    def run_bptt(self, x, y_, alpha, eta, learning=True):
        # feedforward
        t_max = np.shape(x)[0]  
        b_size = np.shape(x)[2]

        gm_in, gm_rec, gm_out = 0, 0, 0
        gpi_rec, gxi_rec = 0, 0

        # input recurrent (feedforward plus recurrent)
        r = np.zeros((t_max+1, self.n_rec, b_size)) 
        h = np.zeros((t_max+1, self.n_rec, b_size))    
        
        u = np.zeros((t_max, self.n_rec, b_size))  
        g = np.zeros((t_max, self.n_rec, b_size))
        delta = np.zeros((t_max, self.n_rec, b_size))
        epsi1 = np.random.normal(0,1,(t_max, self.n_rec, b_size))        

        # single output 
        g_out = np.zeros((self.n_out, b_size))
        delta_out = np.zeros((self.n_out, b_size))
        epsi2 = np.random.normal(0,1,(self.n_out, b_size))                 
        y = np.zeros((self.n_out, b_size))  # RNN output
        err = np.zeros((self.n_out, b_size))  # readout error     
        
        # feedforward  
        self.rescale()
        self.normalization()
        self.update_moment()
        h0 = self.sigma_h0 * np.random.randn(self.n_rec, b_size)
        h[-1] = 1*h0
        r[-1] = self.f(h[-1])           
        for tt in range(-1, t_max-1, 1):
            g[tt+1] = np.dot(self.mu_rec, r[tt]) + np.dot(self.mu_in, x[tt+1])
            delta[tt+1] = 1e-30 + np.sqrt(np.dot(self.rho_rec-self.mu_rec**2, r[tt]**2) \
                                + np.dot(self.rho_in-self.mu_in**2, x[tt+1]**2))              
            u[tt+1] = g[tt+1] + delta[tt+1] * epsi1[tt+1] 
            h[tt+1] = h[tt] + alpha*(-h[tt] + u[tt+1])
            r[tt+1] = self.f(h[tt+1])

        g_out = np.dot(self.mu_out, r[t_max-1])
        delta_out = 1e-30 + np.sqrt(np.dot(self.rho_out-self.mu_out**2, r[t_max-1]**2))       
        y = g_out + delta_out * epsi2

        # backpropagation   
        self.y = y*1
        if not learning:
            return None
        
        err = y - y_                 
        z = np.zeros((t_max, self.n_rec, b_size))
        v = np.zeros((t_max-1, self.n_rec, b_size))
        
        # t = T
        z[t_max-1] = self.df(h[t_max-1]) * np.dot((self.mu_out).T, err) 
        z[t_max-1] += r[t_max-1] * self.df(h[t_max-1]) * np.dot((self.rho_out-self.mu_out**2).T,\
                   err*epsi2/delta_out*self.variance_fix(delta_out))      
        for tt in range(t_max-1, -1, -1):
            if tt > 0:
                z[tt-1] = z[tt]*(1 - alpha)
                v[tt-1] = self.df(h[tt-1])*( np.dot((self.mu_rec).T, z[tt]) \
                            + r[tt-1]*np.dot((self.rho_rec-self.mu_rec**2).T, \
                              z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt])) )  
                z[tt-1] += alpha*v[tt-1]
             
            # gradient for the theta:
            gm_rec += alpha*(1-self.pi_rec)*np.dot(z[tt], r[tt-1].T)+\
                      alpha*self.mu_rec*self.pi_rec\
                      *np.dot(z[tt]*epsi1[tt]/delta[tt]*self.variance_fix(delta[tt]),(r[tt-1]**2).T)
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

        # gradient of output layer               
        gm_out += (1-self.pi_out) * np.dot(err, r[t_max-1].T)\
                  +self.mu_out*self.pi_out\
                  *np.dot(err*epsi2/delta_out*self.variance_fix(delta_out),(r[t_max-1]**2).T)


        # gradient descent
        self.m_in_tr  -= eta*gm_in*self.sigma_in/self.n_in**0.5
        self.m_rec_tr  -= eta*gm_rec*self.sigma_rec/self.n_rec**0.5
        self.xi_rec_tr -= eta*gxi_rec/self.n_rec 
        self.pi_rec_tr -= eta*gpi_rec*1
        self.m_out_tr  -= eta*gm_out*self.sigma_out/self.n_rec**0.5
         
                      
    def rntk_bptt(self, SaS_output=False):           
        # g = (b_size, N, N)
        # rescale m, Xi
        gm_in = self.gm_in*self.sigma_in/self.n_in**0.5
        gm_rec = self.gm_rec*self.sigma_rec/self.n_rec**0.5
        gm_out = self.gm_out*self.sigma_out/self.n_rec**0.5
        gxi_rec = self.gxi_rec/self.n_rec 
        gpi_rec = self.gpi_rec*1
        if SaS_output:
            gxi_out = self.gxi_out/self.n_rec
            gpi_out = self.gpi_out*1
        # RNTK
        # b_size = gm_in.shape[0]
        # rntk = np.zeros((b_size, b_size))
        # for i in range(b_size):
        #     for j in range(i+1):
        #         rntk[i][j] = (gm_in[i]*gm_in[j]).sum() +  (gm_rec[i]*gm_rec[j]).sum()\
        #                     +(gm_out[i]*gm_out[j]).sum() + (gpi_rec[i]*gpi_rec[j]).sum()\
        #                     +(gxi_rec[i]*gxi_rec[j]).sum()
        #         if SaS_output:
        #             rntk[i][j] += (gpi_out[i]*gpi_out[j]).sum() + (gxi_out[i]*gxi_out[j]).sum()
        rntk1 = np.einsum('aij,bij->ab',gm_in,gm_in) + np.einsum('aij,bij->ab',gm_rec,gm_rec)\
              + np.einsum('aij,bij->ab',gm_out,gm_out) + np.einsum('aij,bij->ab',gpi_rec,gpi_rec)\
              + np.einsum('aij,bij->ab',gxi_rec,gxi_rec)
        if SaS_output:
            rntk1 += np.einsum('aij,bij->ab',gpi_out,gpi_out) + np.einsum('aij,bij->ab',gxi_out,gxi_out)
        return rntk1


#%% RNTK
class RNTK:
    def __init__(self, n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0):
        self.n_in = n_in
        self.n_rec = n_rec
        self.n_out = n_out

        # RNTK parameter
        self.sigma_in = sigma_in
        self.sigma_rec = sigma_rec
        self.sigma_out = sigma_out
        self.sigma_h0 = sigma_h0


    # Gaussian Integral of ReLU
    def GI_ReLU(self, t, tp):
        # k1: var of x, k2: var of x', k3: covar of x x'
        # k1
        var_x = np.diagonal(self.Sigma[t][t]).reshape(-1,1)
        vector_one = np.ones_like(var_x)
        k1 = np.dot(var_x, vector_one.T)
        # k2
        var_xp = np.diagonal(self.Sigma[tp][tp]).reshape(-1,1)
        vector_one = np.ones_like(var_xp)       
        k2 = np.dot(vector_one, var_xp.T)
        # k3
        k3 = 1*self.Sigma[t][tp]
        c = k3/np.sqrt(k1*k2)
        f = 1/(2*np.pi)*(c*(np.pi-np.arccos(c)) + np.sqrt(1-c**2))*np.sqrt(k1*k2)
        return f


    # Gaussian Integral of dReLU
    def GI_dReLU(self, t, tp):
        # k1: var of x, k2: var of x', k3: covar of x x'
        # k1
        var_x = np.diagonal(self.Sigma[t][t]).reshape(-1,1)
        vector_one = np.ones_like(var_x)
        k1 = np.dot(var_x, vector_one.T)
        # k2
        var_xp = np.diagonal(self.Sigma[tp][tp]).reshape(-1,1)
        vector_one = np.ones_like(var_xp)       
        k2 = np.dot(vector_one, var_xp.T)
        # k3
        k3 = self.Sigma[t][tp]
        c = k3/np.sqrt(k1*k2)
        f = 1/(2*np.pi)*(np.pi-np.arccos(c))
        return f
    
    
    # E[u^t(x), u^t'(x')]
    def Omega_func(self, x, t, tp):
        # x: (t_max, N_in, b_size)
        b_size = x.shape[2]
        input_term = self.sigma_in**2/self.n_in*np.dot(x[t].T, x[tp])
        gaussint = self.GI_ReLU(t-1, tp-1)
        # t=t': give identity
        if t == tp:
            gi_term = (self.sigma_rec**2 + 1/2*np.identity(b_size))*gaussint
        else:
            gi_term = self.sigma_rec**2*gaussint
        Omega = gi_term + input_term
        return Omega
    
    
    # E[h^t(x), h^t'(x')]
    def Sigma_func(self, alpha, t, tp):
        b_size = (self.Sigma[t][tp]).shape[0]
        if (t!=-1 and tp!=-1):
            Sigma_ttp = (1-alpha)**2*self.Sigma[t-1][tp-1] + alpha**2*self.Omega[t][tp]
            for dt in range(1, tp+1):
                Sigma_ttp += (1-alpha)**dt*alpha**2*self.Omega[t][tp-dt]
            for dt in range(1, t+1):
                Sigma_ttp += (1-alpha)**dt*alpha**2*self.Omega[t-dt][tp] 
        else:
            Sigma_ttp = (1-alpha)**(t+1 + tp+1)*np.identity(b_size)*self.sigma_h0**2   
        return Sigma_ttp
    

    # E[v^t(x), v^t'(x')]
    def Gamma_func(self, t, tp):
        Gamma = self.sigma_rec**2*self.GI_dReLU(t, tp)*self.Pi[t+1][tp+1]
        return Gamma   
    

    # E[delta^t(x), delta^t'(x')]
    def Pi_func(self, alpha, t, tp):
        t_max = self.Pi.shape[0]
        if ( t!=(t_max-1) and (tp!=(t_max-1)) ):
            Pi_ttp = (1-alpha)**2*self.Pi[t+1][tp+1] + alpha**2*self.Gamma[t][tp]
            for dt in range(1, t_max-tp-1):
                Pi_ttp += (1-alpha)**dt*alpha**2*self.Gamma[t][tp+dt]
            for dt in range(1, t_max-t -1):
                Pi_ttp += (1-alpha)**dt*alpha**2*self.Gamma[t+dt][tp] 
        else:
            Pi_ttp = (1-alpha)**(t_max-1-t + t_max-1-tp)*self.Pi[t_max-1][t_max-1]
        return Pi_ttp
    
    
    def GP_kernel(self, x, alpha):
        # x = (t_max, N_in, b_size)
        t_max = x.shape[0]
        b_size = x.shape[2]
        
        ## Forward pass
        # Sigma: kernel of pre-activation
        # Sigma[-1][-1] -> E[h_0,h_0]
        self.Sigma = np.zeros((t_max+1, t_max+1, b_size, b_size))
        self.Omega = np.zeros((t_max, t_max, b_size, b_size))
        for t in range(-1, t_max, 1):    
            # compute Omega(t, t')
            for t_in in range(0, t+1):
                if t_in == t:
                    self.Omega[t][t] = self.Omega_func(x, t, t)
                else:
                    self.Omega[t_in][t] = self.Omega_func(x, t_in, t)
                    self.Omega[t][t_in] = self.Omega_func(x, t, t_in)
            
            # compute Sigma(t, t')
            for t_in in range(-1, t+1):
                if t_in == t:
                    self.Sigma[t][t] = self.Sigma_func(alpha, t, t)      
                else:
                    self.Sigma[t][t_in] = self.Sigma_func(alpha, t, t_in)     
                    self.Sigma[t_in][t] = self.Sigma_func(alpha, t_in, t)     
        
        ## Backward pass
        # Pi: kernel of error
        self.Pi = np.zeros((t_max, t_max, b_size, b_size))
        self.Gamma = np.zeros((t_max-1, t_max-1, b_size, b_size))
        self.Pi[t_max-1][t_max-1] = self.sigma_out**2 * self.GI_dReLU(t_max-1, t_max-1)
        for t in range(t_max-1, -1, -1):
            # compute Gamma (t, t')
            for t_in in range(t, t_max-1):
                if t_in == t:
                    self.Gamma[t][t] = self.Gamma_func(t, t)
                else:
                    self.Gamma[t_in][t] = self.Gamma_func(t_in, t)
                    self.Gamma[t][t_in] = self.Gamma_func(t, t_in)
                    
            # compute Pi (t, t')
            for t_in in range(t, t_max):
                if t_in == t:
                    self.Pi[t][t] = self.Pi_func(alpha, t, t)
                else:
                    self.Pi[t_in][t] = self.Pi_func(alpha, t_in, t)
                    self.Pi[t][t_in] = self.Pi_func(alpha, t, t_in)       
        
        
    def compute_rntk(self, x, alpha, SaS_output=False):
        # x = (t_max, N_in, b_size)
        t_max = x.shape[0]
        rntk_m_in = 0
        rntk_m_rec = 0
        rntk_m_out = 0
        rntk_pi_rec = 0
        rntk_xi_rec = 0
        for t in range(0, t_max):
            for tp in range(0, t_max):
                rntk_m_rec += self.Pi[t][tp]*alpha**2*self.sigma_rec**2*self.GI_ReLU(t-1, tp-1)
                rntk_pi_rec += self.Pi[t][tp]*alpha**2*(self.sigma_rec**2*self.GI_ReLU(t-1, tp-1))
                rntk_m_in += self.Pi[t][tp]*alpha**2*self.sigma_in**2/self.n_in*np.dot(x[t].T,x[tp])
        rntk_xi_rec = np.zeros(np.shape(rntk_m_rec)) 
        rntk_m_out = self.sigma_out**2*self.GI_ReLU(t_max-1, t_max-1)
        if SaS_output:
            # epsi_out = np.random.randn(b_size, 1)
            rntk_pi_out = self.sigma_out**2*self.GI_ReLU(t_max-1, t_max-1)
            rntk_xi_out = np.zeros(np.shape(rntk_pi_out))
            self.rntk = rntk_m_in + rntk_m_rec + rntk_m_out + rntk_pi_rec + rntk_xi_rec + rntk_pi_out + rntk_xi_out
            return rntk_m_in, rntk_m_rec, rntk_m_out, rntk_pi_rec, rntk_xi_rec, rntk_pi_out, rntk_xi_out
        self.rntk = rntk_m_in + rntk_m_rec + rntk_m_out + rntk_pi_rec + rntk_xi_rec
        return rntk_m_in, rntk_m_rec, rntk_m_out, rntk_pi_rec, rntk_xi_rec


    def GP_kernel_dynamics(self, x):
        # alpha
        alpha = 1
        
        # x = (t_max, N_in, b_size)
        t_max = x.shape[0]
        b_size = x.shape[2]
        
        ## Forward pass
        # Sigma: kernel of pre-activation
        # Sigma[-1][-1] -> E[h_0,h_0]
        self.Sigma = np.zeros((t_max+1, t_max+1, b_size, b_size))
        self.Sigma[-1][-1] = self.Sigma_func(alpha, -1, -1) 
        for t in range(0, t_max, 1):    
            self.Sigma[t][t] = self.Omega_func(x, t, t)      
        
        ## Backward pass
        # Pi: kernel of error
        self.Pi = np.zeros((t_max, t_max, b_size, b_size))
        self.Pi[t_max-1][t_max-1] = self.sigma_out**2 * self.GI_dReLU(t_max-1, t_max-1)
        for t in range(t_max-2, -1, -1):
            self.Pi[t][t] = self.Gamma_func(t, t)
        

    def compute_rntk_dynamics(self, x):
        # alpha 
        alpha = 1
        # x = (t_max, N_in, b_size)
        t_max = x.shape[0]
        rntk_m_in = 0
        rntk_m_rec = 0
        rntk_m_out = 0
        rntk_pi_rec = 0
        rntk_xi_rec = 0
        for t in range(0, t_max):
            rntk_m_rec += self.Pi[t][t]*alpha**2*self.sigma_rec**2*self.GI_ReLU(t-1, t-1)
            rntk_pi_rec += self.Pi[t][t]*alpha**2*(self.sigma_rec**2*self.GI_ReLU(t-1, t-1))
            rntk_xi_rec += np.zeros(rntk_pi_rec.shape)
            rntk_m_in += self.Pi[t][t]*alpha**2*self.sigma_in**2/self.n_in*np.dot(x[t].T,x[t])
        rntk_m_out = self.sigma_out**2*self.GI_ReLU(t_max-1, t_max-1)
        self.rntk = rntk_m_in + rntk_m_rec + rntk_m_out + rntk_pi_rec + rntk_xi_rec
        return rntk_m_in, rntk_m_rec, rntk_m_out, rntk_pi_rec, rntk_xi_rec



#%% Test
if __name__ ==  '__main__':
    # parameter
    alpha = 0.9
    n_in = 1
    n_rec = 2000
    n_out = 1
    phi = 4.5
    sigma_in = sigma_rec = sigma_out = 1
    sigma_h0 = 1
    
    # x = (t_max, n_in, b_size)
    def x_func(phi):
        x1 = np.array([1, -1], ndmin=2).T
        x2 = np.array([np.cos(phi), np.sin(phi)], ndmin=2).T
        x = np.concatenate((x1, x2), axis=1)
        x = np.expand_dims(x, 1)
        return x
    x = x_func(phi)
    b_size = x.shape[2]
    t_max = x.shape[0]
    
    # network
    SaS_output = False
    network1 = RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    kernel1 = RNTK(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    if SaS_output:
        network1.SaS_output()
    
    
    #%%% Test 1: Analytical K
    # # Cov
    # Sigma = (np.array([[1,0.3],[0.3,0.5]])).reshape(1,1,2,2)
    # t = tp = 0
    
    # # analytical
    # kernel1.Sigma = Sigma*1
    # Kphi1 = kernel1.GI_ReLU(t, tp)
    # Kphip1 = kernel1.GI_dReLU(t, tp)
    
    
    # # monte carlo
    # f = lambda x: np.maximum(0, x)
    # df = lambda x: np.where(x>0, 1, 0)
    # z = np.random.multivariate_normal([0,0], Sigma[t][tp], 2000)
    # phi2, phip2 = f(z), df(z)
    # Kphi2 = 1/2000*np.dot(phi2.T, phi2)
    # Kphip2 = 1/2000*np.dot(phip2.T, phip2)

    
    #%%% Test 2: Sigma & Omega matrix    
    # # analytical
    # kernel1.GP_kernel(x, alpha)
    # Sigma1s = kernel1.Sigma
    # Omega1s = kernel1.Omega
    
    # # monte carlo
    # check = "Sigma"
    # rep = 10
    # us = np.zeros([rep, t_max, n_rec, b_size])
    # hs = np.zeros([rep, t_max+1, n_rec, b_size])
    # for i in range(rep):
    #     u, h = network1.gradient_bptt(x, alpha, check, SaS_output)
    #     us[i], hs[i] = u*1, h*1
        
    # Sigma2s = (np.einsum('btnx,bsny->btsxy',hs,hs)/n_rec)[:,:,:,:,:]
    # Omega2s = (np.einsum('btnx,bsny->btsxy',us,us)/n_rec)[:,:,:,:,:]
    
    
    # #%%%% plot
    # Sigma1_1d = Sigma1s.reshape(-1)
    # Sigma2_1d = Sigma2s.reshape(rep,-1)
    # Sigma2_mean, Sigma2_std = (Sigma2s.mean(0)).reshape(-1), (Sigma2s.std(0)).reshape(-1)
    
    # Omega1_1d = Omega1s.reshape(-1)
    # Omega2_1d = Omega2s.reshape(rep,-1)
    # Omega2_mean, Omega2_std = (Omega2s.mean(0)).reshape(-1), (Omega2s.std(0)).reshape(-1)
    
    
    # plt.figure(figsize=(18,8))
    # plt.subplot(121)
    # y = np.arange(0.1,1.2,0.01)
    # plt.plot(y,y, ls='--', c='k', alpha=0.5, lw=4, label=r"y=x")
    # plt.errorbar(Sigma1_1d, Sigma2_mean, yerr=Sigma2_std, color='k', lw=0,\
    #               marker='D', ms=10, mfc='none', mew=2, elinewidth=4, capsize=10, capthick=4)
    # plt.legend(fontsize=25, loc=4)
    # plt.xlabel(r"$\Sigma^{t,t^{\prime}}(x,x)$ (Analytical)", size=25)
    # plt.ylabel(r"$\Sigma^{t,t^{\prime}}(x,x)$ (BPTT)", size=25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    
    # plt.subplot(122)
    # plt.plot(y,y, ls='--', c='k', alpha=0.5, lw=4, label=r"y=x")
    # plt.errorbar(Omega1_1d, Omega2_mean, yerr=Omega2_std, color='k', lw=0,\
    #               marker='D', ms=10, mfc='none', mew=2, elinewidth=4, capsize=10, capthick=4)
    # plt.legend(fontsize=25, loc=4)
    # plt.xlabel(r"$\Omega^{t,t^{\prime}}(x,x)$ (Analytical)", size=25)
    # plt.ylabel(r"$\Omega^{t,t^{\prime}}(x,x)$ (BPTT)", size=25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()
    
    
    #%%% Test 3: Pi matrix
    # # analytical
    # kernel1.GP_kernel(x, alpha)
    # Pi1s = kernel1.Pi
    # Gamma1s = kernel1.Gamma
    
    # # monte carlo
    # check = "Pi"
    # rep = 20
    # deltas = np.zeros([rep, t_max, n_rec, b_size])
    # vs = np.zeros([rep, t_max-1, n_rec, b_size])
    # for i in range(rep):
    #     delta, v = network1.gradient_bptt(x, alpha, check)
    #     deltas[i] = delta*1
    #     vs[i] = v*1
    #     print("\r{}".format(i), end='')
    # Pi2s = np.einsum('btrx,bsry->btsxy', deltas, deltas)
    # Gamma2s = np.einsum('btrx,bsry->btsxy', vs, vs)
    
    # #%%%% plot
    # Pi1_1d = Pi1s.reshape(-1)
    # Pi2_1d = Pi2s.reshape(rep,-1)
    # Pi2_mean, Pi2_std = (Pi2s.mean(0)).reshape(-1), (Pi2s.std(0)).reshape(-1)
    
    # Gamma1_1d = Gamma1s.reshape(-1)
    # Gamma2_1d = Gamma2s.reshape(rep,-1)
    # Gamma2_mean, Gamma2_std = (Gamma2s.mean(0)).reshape(-1), (Gamma2s.std(0)).reshape(-1)
    
    
    # plt.figure(figsize=(18,8))
    # plt.subplot(121)
    # y = np.arange(0,0.6,0.01)
    # plt.plot(y,y, ls='--', c='k', alpha=0.5, lw=4, label=r"y=x")
    # plt.errorbar(Pi1_1d, Pi2_mean, yerr=Pi2_std, color='k', lw=0,\
    #               marker='D', ms=10, mfc='none', mew=2, elinewidth=4, capsize=10, capthick=4)
    # plt.legend(fontsize=25, loc=4)
    # plt.xlabel(r"$\Pi^{t,t^{\prime}}(x,x)$ (Analytical)", size=25)
    # plt.ylabel(r"$\Pi^{t,t^{\prime}}(x,x)$ (BPTT)", size=25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    
    # plt.subplot(122)
    # plt.plot(y,y, ls='--', c='k', alpha=0.5, lw=4, label=r"y=x")
    # plt.errorbar(Gamma1_1d, Gamma2_mean, yerr=Gamma2_std, color='k', lw=0,\
    #               marker='D', ms=10, mfc='none', mew=2, elinewidth=4, capsize=10, capthick=4)
    # plt.legend(fontsize=25, loc=4)
    # plt.xlabel(r"$\Gamma^{t,t^{\prime}}(x,x)$ (Analytical)", size=25)
    # plt.ylabel(r"$\Gamma^{t,t^{\prime}}(x,x)$ (BPTT)", size=25)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()
    
    #%%% Test 4: gradient
    # # analytical
    # kernel1.GP_kernel(x, alpha)
    # if SaS_output:
    #     rntk_m_in1, rntk_m_rec1, rntk_m_out1, rntk_pi_rec1, rntk_xi_rec1, rntk_pi_out1, rntk_xi_out1 = kernel1.compute_rntk(x, alpha, SaS_output)
    # else:
    #     rntk_m_in1, rntk_m_rec1, rntk_m_out1, rntk_pi_rec1, rntk_xi_rec1 = kernel1.compute_rntk(x, alpha)

    # ## monte carlo
    # network1.gradient_bptt(x, alpha, SaS_output=SaS_output)
    # gm_in = network1.gm_in*network1.sigma_in/network1.n_in**0.5
    # gm_rec = network1.gm_rec*network1.sigma_rec/network1.n_rec**0.5
    # gm_out = network1.gm_out*network1.sigma_out/network1.n_rec**0.5
    # gxi_rec = network1.gxi_rec/network1.n_rec 
    # gpi_rec = network1.gpi_rec
    # if SaS_output:
    #     gxi_out = network1.gxi_out/network1.n_rec
    #     gpi_out = network1.gpi_out*1
    
    # # component    
    # x_index = 1
    # rntk_m_in = (gm_in[x_index]*gm_in[x_index]).sum()
    # rntk_m_rec = (gm_rec[x_index]*gm_rec[x_index]).sum()
    # rntk_m_out = (gm_out[x_index]*gm_out[x_index]).sum()
    # rntk_pi_rec = (gpi_rec[x_index]*gpi_rec[x_index]).sum()
    # rntk_xi_rec = (gxi_rec[x_index]*gxi_rec[x_index]).sum()
    # if SaS_output:
    #     rntk_pi_out = (gpi_out[x_index]*gpi_out[x_index]).sum()
    #     rntk_xi_out = (gxi_out[x_index]*gxi_out[x_index]).sum()
    
    
    # #%%% Test 5: RNTK 
    # # bptt kernel
    # network1 = RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    # network1.gradient_bptt(x, alpha, SaS_output=SaS_output)
    # rntk1 = network1.rntk_bptt(SaS_output)
    
    # # analytical kernel
    # kernel1 = RNTK(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    # kernel1.GP_kernel(x, alpha)
    # kernel1.compute_rntk(x, alpha, SaS_output)  
    # rntk2 = kernel1.rntk
    
    # #%%% Test 6: alpha = 1
    # # bptt kernel
    # network1 = RNN(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    # network1.gradient_bptt(x, 1)
    # rntk1 = network1.rntk_bptt()
    
    # kernel1 = RNTK(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    # kernel2 = RNTK(n_in, n_rec, n_out, sigma_in, sigma_rec, sigma_out, sigma_h0)
    # # arbitrary alpha 
    # kernel1.GP_kernel(x, 1)
    # kernel1.compute_rntk(x, 1, False)  
    # rntk_alpha1 = kernel1.rntk
   
    # # alpha = 1
    # kernel2.GP_kernel_dynamics(x)
    # kernel2.compute_rntk_dynamics(x)  
    # rntk_alpha2 = kernel2.rntk
