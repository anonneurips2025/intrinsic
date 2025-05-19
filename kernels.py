#!/usr/bin/env python
# coding: utf-8



import numpy as np


from scipy import special
import math

def gaussian_kernel(x,y,diagonal=0):
    dim = x.shape[1]
    if diagonal==0:
        x1 = np.reshape(np.sum(np.square(x),axis=1), [-1,1])
        y1 = np.reshape(np.sum(np.square(y),axis=1), [1,-1])
        x2 = x1+y1-2*np.matmul(x,np.transpose(y))
        return np.exp(-x2/dim)
    if diagonal==1:
        size = x.shape[0]
        return np.ones(size)
    return np.exp(-np.sum(np.square(x-y))/dim)

def laplace_kernel(x,y,diagonal=0):
    if diagonal==0:
        x1 = np.reshape(np.sum(np.square(x),axis=1), [-1,1])
        y1 = np.reshape(np.sum(np.square(y),axis=1), [1,-1])
        x2 = x1+y1-2*np.matmul(x,np.transpose(y))
        return np.exp(-np.sqrt(x2+1e-10))
    if diagonal==1:
        size = x.shape[0]
        return np.ones(size)
    return np.exp(-np.sqrt(np.sum(np.square(x-y))))

def one_over_two_kernel(x,y,diagonal=0):
    if diagonal==0:
        x1 = np.reshape(np.sum(np.square(x),axis=1), [-1,1])
        y1 = np.reshape(np.sum(np.square(y),axis=1), [1,-1])
        x2 = x1+y1-2*np.matmul(x,np.transpose(y))
        return np.exp(-np.power(x2+1e-10, 0.25))
    if diagonal==1:
        size = x.shape[0]
        return np.ones(size)
    return np.exp(-np.power(np.sum(np.square(x-y)),0.25))

def euclid_kernel(x,y,diagonal=0):
    if diagonal==0:
        return np.matmul(x,np.transpose(y))
    if diagonal==1:
        return np.sum(np.square(x),axis=1)
    return 0.1

ReLU = lambda x: x * (x > 0)
exp = lambda x: np.exp(-0.5*np.multiply(x,x))
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
erf = lambda x: special.erf(x)

ReLU_prime = lambda x: (x > 0)*1.0
exp_prime = lambda x: -np.multiply(x,np.exp(-0.5*np.multiply(x,x)))
sigmoid_prime = lambda x: 1/np.multiply(1 + np.exp(-x),1 + np.exp(x))
tanh_prime = lambda x: np.square(2 / (np.exp(x) + np.exp(-x)))
erf_prime = lambda x: (2/np.sqrt(math.pi))*np.exp(-np.multiply(x,x))

M = 1000
features = np.random.normal(loc=0.0, scale=1.0, size=(15*M))
    
def nngp1_kernel(x,y,activation,diagonal=0):
    dim = x.shape[1]
    feats = np.reshape(features[:dim*M], [dim,M])
    if diagonal==0:
        x_feat = activation(np.matmul(x,feats))
        y_feat = activation(np.matmul(y,feats))
        ker_m = np.matmul(x_feat,y_feat.T)/M
        if x.shape[0] == y.shape[0]:
            return ker_m+1e-10*np.identity(x.shape[0])
        return ker_m
    if diagonal==1:
        size = x.shape[0]
        x_feat = activation(np.matmul(x,feats))
        return np.mean(np.square(x_feat),axis=1)
    return 0.1

def nngp1_relu(x,y,diagonal=0):
    return nngp1_kernel(x,y,ReLU,diagonal)
def nngp1_exp(x,y,diagonal=0):
    return nngp1_kernel(x,y,exp,diagonal)
def nngp1_sigmoid(x,y,diagonal=0):
    return nngp1_kernel(x,y,sigmoid,diagonal)
def nngp1_tanh(x,y,diagonal=0):
    return nngp1_kernel(x,y,tanh,diagonal)
def nngp1_erf(x,y,diagonal=0):
    return nngp1_kernel(x,y,erf,diagonal)

features2 = np.random.normal(loc=0.0, scale=1.0, size=(M,M))/np.sqrt(M)
def nngp2_kernel(x,y,activation1,activation2,diagonal=0):
    dim = x.shape[1]
    feats = np.reshape(features[:dim*M], [dim,M])
    if diagonal==0:
        x_feats = activation1(np.matmul(x,feats))
        x_feats = activation2(np.matmul(x_feats, features2))
        y_feats = activation1(np.matmul(y,feats))
        y_feats = activation2(np.matmul(y_feats, features2))
        ker_m = np.matmul(x_feats,y_feats.T)/M
        if x.shape[0] == y.shape[0]:
            return ker_m+1e-10*np.identity(x.shape[0])
        return ker_m
    if diagonal==1:
        size = x.shape[0]
        x_feats = activation1(np.matmul(x,feats))
        x_feats = activation2(np.matmul(x_feats, features2))
        return np.mean(np.square(x_feats),axis=1)
    return 0.1

def nngp2_relu(x,y,diagonal=0):
    return nngp2_kernel(x,y,ReLU,ReLU,diagonal)
def nngp2_exp(x,y,diagonal=0):
    return nngp2_kernel(x,y,exp,exp,diagonal)
def nngp2_sigmoid(x,y,diagonal=0):
    return nngp2_kernel(x,y,sigmoid,sigmoid,diagonal)
def nngp2_tanh(x,y,diagonal=0):
    return nngp2_kernel(x,y,tanh,tanh,diagonal)
def nngp2_erf(x,y,diagonal=0):
    return nngp2_kernel(x,y,erf,erf,diagonal)

def ntk1_kernel(x,y,activation1,activation2,diagonal=0):
    a = np.multiply(nngp1_kernel(x,y,activation2,diagonal), euclid_kernel(x,y,diagonal))
    return a+nngp1_kernel(x,y,activation1,diagonal)
def ntk1_relu(x,y,diagonal=0):
    return ntk1_kernel(x,y,ReLU,ReLU_prime,diagonal)
def ntk1_exp(x,y,diagonal=0):
    return ntk1_kernel(x,y,exp,exp_prime,diagonal)
def ntk1_sigmoid(x,y,diagonal=0):
    return ntk1_kernel(x,y,sigmoid,sigmoid_prime,diagonal)
def ntk1_tanh(x,y,diagonal=0):
    return ntk1_kernel(x,y,tanh,tanh_prime,diagonal)
def ntk1_erf(x,y,diagonal=0):
    return ntk1_kernel(x,y,erf,erf_prime,diagonal)

def ntk2_kernel(x,y,activation1,activation2,diagonal=0):
    dot_1 = nngp1_kernel(x,y,activation2,diagonal)
    dot_2 = nngp2_kernel(x,y,activation1,activation2,diagonal)
    a = np.multiply(dot_2,np.multiply(dot_1, euclid_kernel(x,y,diagonal)))
    b = np.multiply(dot_2,nngp1_kernel(x,y,activation1,diagonal))
    c = nngp2_kernel(x,y,activation1,activation1,diagonal)
    return a+b+c

def ntk2_relu(x,y,diagonal=0):
    return ntk2_kernel(x,y,ReLU,ReLU_prime,diagonal)
def ntk2_exp(x,y,diagonal=0):
    return ntk2_kernel(x,y,exp,exp_prime,diagonal)
def ntk2_sigmoid(x,y,diagonal=0):
    return ntk2_kernel(x,y,sigmoid,sigmoid_prime,diagonal)
def ntk2_tanh(x,y,diagonal=0):
    return ntk2_kernel(x,y,tanh,tanh_prime,diagonal)
def ntk2_erf(x,y,diagonal=0):
    return ntk2_kernel(x,y,erf,erf_prime,diagonal)
