import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal
from scipy.special import digamma, gamma
import cv2
import data_loader as loader
import sys
IMAGE_SIZE = 10

def MLE(x):
    x = np.array(x)
    mean = np.reshape(np.sum(x,axis=0)/len(x), (IMAGE_SIZE,IMAGE_SIZE))
    covar = np.diag(np.sum((x-mean.flatten())**2,axis=0)/len(x))
    
    return mean, covar

class GMM(object):
    def __init__(self, X, k=2):
        # dimension
        X = np.array(X)
        # X = X[:,:60*60]
        self.m, self.n = X.shape
        self.data = X.copy()
        # number of mixtures
        self.k = k
        
    def _init(self):
        # init mixture means/sigmas
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)])
        self.lambda_val = np.ones(self.k)/self.k
        self.r = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        _,data_variance = MLE(self.data)
        for i in range(self.k):
        	self.sigma_arr[i] = data_variance
        	self.mean_arr[i] = self.data[i+10]

    def fit(self, tol=1e-1):
        self._init()
        num_iters = 0
        while(num_iters<20):
            self.E_step()
            self.M_step()
            num_iters += 1
#            print('Iteration %d done.'%(num_iters))
            sigma = np.zeros((self.k,self.n,self.n))
            mean = np.zeros((self.k,IMAGE_SIZE,IMAGE_SIZE))
        for i in range(self.k):
            mean[i] = np.reshape(self.mean_arr[i],(IMAGE_SIZE,IMAGE_SIZE))
        
        return mean, self.sigma_arr, self.lambda_val    

    def E_step(self):
        total_den = np.zeros((self.m,1))
        for i in range(self.k):
            self.r[:,i] = (self.lambda_val[i]*multivariate_normal.pdf(self.data, self.mean_arr[i].A1, self.sigma_arr[i])).reshape(-1,1)
            total_den = np.add(total_den,self.r[:,i])
        self.r = np.divide(self.r,total_den)

    def M_step(self):
        total_r = np.sum(np.sum(self.r))
        
        for j in range(self.k):
	        const = np.sum(self.r[:,j])
	        self.lambda_val[j] = const/total_r
	        self.mean_arr[j] = np.matmul(self.r[:,j].T,self.data)/const
	        x_sub_u = self.data - self.mean_arr[j]
	        r_x_sub_u = np.multiply(self.r[:,j].flatten(),x_sub_u.T)
	        r_x_sub_u = r_x_sub_u.T
	        self.sigma_arr[j] = np.diag(np.diag(np.matmul(r_x_sub_u.T, x_sub_u)/const))
#        print('Lambda Value:',self.lambda_val)

    def test(self,mean_face, covar_face, lambda_val_face, mean_nonface, covar_nonface, lambda_val_nonface):
        self.mean_face = np.zeros((self.k, self.n))
        self.mean_nonface = np.zeros((self.k, self.n))
        
        for i in range(self.k):
            self.mean_face[i] = mean_face[i].flatten()
            self.mean_nonface[i] = mean_nonface[i].flatten()
        print(self.mean_face.shape, covar_face.shape)
        
        p1 = np.zeros(self.m)
        p2 = np.zeros(self.m)
        
        for i in range(self.k):
            p1 = p1 + lambda_val_face[i]*multivariate_normal.pdf(self.data, self.mean_face[i], covar_face[i])
            p2 = p2 + lambda_val_nonface[i]*multivariate_normal.pdf(self.data, self.mean_nonface[i], covar_nonface[i])
        p1[p1==0] = 1e-213
        return np.log(p1), np.log(p2)


class t_Distribution(object):
    def __init__(self, X, V):
        self.x = np.array(X)
        self.m, self.n = self.x.shape
        self.data = np.copy(self.x)
        self.v = V
        
    def _init(self):
        self.mean,self.sigma = MLE(self.data)
        self.mean = self.mean.flatten()
        self.E_h = np.zeros(self.m)
        self.E_log_h = np.zeros(self.m)
        
    def fit(self):
        self._init()
        num_iters = 0
        while(num_iters<20):
            self.E_step()
            self.M_step()
            num_iters += 1
#            print('Iteration %d done.'%(num_iters))
            mean = np.reshape(self.mean,(IMAGE_SIZE,IMAGE_SIZE))
        
        return mean, self.sigma
    
    def E_step(self):
        sigma_inv = inv(self.sigma)
        x_sub_u = self.data - self.mean
        term1 = np.matmul(x_sub_u,sigma_inv)
        term2 = np.sum(np.multiply(term1,x_sub_u), axis = 1)
        term = (self.v + term2)
        
        self.E_h = (self.v + self.n)/term
        self.E_log_h = digamma((self.v + self.n)/2) - np.log(term/2)
        
    def M_step(self):
        self.mean = np.matmul(self.E_h,self.data)/np.sum(self.E_h)
        x_sub_u = self.data - self.mean
        term2 = np.multiply(x_sub_u,x_sub_u)
        self.sigma = np.diag(np.matmul(self.E_h, term2)/np.sum(self.E_h))
        
    def test(self,mean_face, covar_face, mean_nonface, covar_nonface):
        mean_face = mean_face.flatten()
        mean_nonface = mean_nonface.flatten()
        
        x_sub_u = self.data - mean_face
        term1 = np.matmul(x_sub_u,inv(covar_face))
        term2 = np.sum(np.multiply(term1,x_sub_u),axis = 1)
        term = 1 + term2/self.v
        log_pdf_face = -np.log(det(np.sqrt(covar_face))) -0.5*(self.v+self.n)*np.log(term)
        
        x_sub_u = self.data - mean_nonface
        term1 = np.matmul(x_sub_u,inv(covar_nonface))
        term2 = np.sum(np.multiply(term1,x_sub_u),axis = 1)
        term = 1 + term2/self.v
        log_pdf_nonface = -np.log(det(np.sqrt(covar_nonface))) -0.5*(self.v+self.n)*np.log(term)
        
        return log_pdf_face, log_pdf_nonface


class Factor_Analysis(object):
    def __init__(self, X, K):
        self.x = np.array(X)
        self.m, self.n = self.x.shape
        self.data = np.copy(self.x)
        self.k = K
        
    def _init(self):
        self.mean,self.sigma = MLE(self.data)
        self.mean = self.mean.flatten()
        self.phi = 1*np.random.random((self.n, self.k))
        self.E_h = np.zeros((self.m, self.k))
        self.E_hhT = np.zeros((self.m, self.k, self.k))
        
    def fit(self):
        self._init()
        num_iters = 0
        while(num_iters<2):
            self.E_step()
            self.M_step()
            num_iters += 1
#            print('Iteration %d done.'%(num_iters))
        mean = np.reshape(self.mean,(IMAGE_SIZE,IMAGE_SIZE))
        
        return mean, self.sigma, self.phi
    
    def E_step(self):
        sigma_inv = inv(self.sigma)
        term1 = np.matmul(np.matmul(self.phi.T,sigma_inv),self.phi)
        term1_inv = inv(term1 + np.identity(self.k))
        term = np.matmul(np.matmul(term1_inv, self.phi.T), sigma_inv)
        
        x_sub_u = self.data - self.mean
        for i in range(self.m):
            self.E_h[i] = np.matmul(term,x_sub_u[i].T)
            self.E_hhT[i] = term1_inv + np.matmul(self.E_h[i], self.E_h[i].T)
            
    def M_step(self):
        term2 = self.E_hhT[0]
        x_sub_u = self.data - self.mean
        for i in range(1,self.m):
            term2 = term2 + self.E_hhT[i]
            
        term1 = np.matmul(x_sub_u.T, self.E_h)
        self.phi = np.matmul(term1,inv(term2))
        
        term = np.zeros(self.n)
        for i in range(self.m):
            term1 = np.matmul(x_sub_u[i].reshape((-1,1)), x_sub_u[i].reshape((1,-1)))
            term2 = np.matmul(np.matmul(self.phi, self.E_h[i]).reshape((-1,1)),x_sub_u[i].reshape((1,-1)))
            term = term + np.diag(term1 - term2)
        self.sigma = np.diag(term/self.m)
        
    def test(self, mean_face, covar_face, phi_face, mean_nonface, covar_nonface, phi_nonface):
        p1 = multivariate_normal.logpdf(self.data, mean_face.flatten(), np.matmul(phi_face,phi_face.T) + abs(covar_face))
        p2 = multivariate_normal.logpdf(self.data, mean_nonface.flatten(), np.matmul(phi_nonface,phi_nonface.T) + abs(covar_nonface))
        
        return p1,p2
