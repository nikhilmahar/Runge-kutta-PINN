# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:44:42 2022

@author: General
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
k=10
m=1
c=0.2

def dU_dx(U, x):
    # Here U is a vector such that y=U[0] and z=U[1]. This function should return [y', z']
    return [U[1], -c*U[1] - k*U[0] +np.sin(10*x)]
    # return [U[1], -c*U[1] - k*U[0] +0]
U0 = [1, 0]
xs = np.linspace(0, 10, 10000)
Us = odeint(dU_dx, U0, xs)
ys = Us[:,0]

# acc=np.sin(10*xs)-(k/m)*ys-(c/m)*Us[:,1]
acc=0-(k/m)*ys-(c/m)*Us[:,1]

plt.xlabel("x")
plt.ylabel("y")
plt.title("disp")
plt.plot(xs,ys);
plt.show()
plt.xlabel("x")
plt.ylabel("y")
plt.title("acc")
plt.plot(xs,acc);
plt.show()

# =============================================================================
# PINN START HERE : INVERSE PROBLEM
# =============================================================================
import torch
import torch.nn as nn
from time import perf_counter
from PIL import Image
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import requests
import os
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import StepLR

device = torch.device("cpu")
K=torch.tensor(k).float()
M=torch.tensor(m).float()
C=torch.tensor(c).float()
DISP=torch.tensor(ys).reshape(len(ys),1).float()
VEL=torch.tensor(Us[:,1]).reshape(len(ys),1).float()
ACC=torch.tensor(acc).reshape(len(acc),1).float()

def Psi(D):
    STIFF=N(D)
    #print("XK",XK)
    return  STIFF
# The loss function
def loss(D,V,A,M,K,C):
    #print("FORCE",FORCE.shape)
    D.requires_grad = True
    XS=torch.tensor(xs).reshape(len(xs),1)
    out = Psi(D)
    # F=torch.sin(XS)
    F=0
   # print("out",out)
    # print("out",out[:,1].shape)
    # Psi_t = torch.autograd.grad(out, TX, grad_outputs=torch.ones_like(out),
    #                               create_graph=True)[0]
    #print("PSI_T",Psi_t.shape)
    # Psi_tt = torch.autograd.grad(Psi_t, TX, grad_outputs=torch.ones_like(Psi_t),
    #                               create_graph=True)[0]
    #print("Psi_tt",Psi_tt[:,0].shape)
    #print((out[:,0].reshape(len(out),1)*out[:,1].reshape(len(out),1)))
    # loss=(torch.matmul(Psi_tt[:,0].reshape(len(out),1),M)+(out[:,0].reshape(len(out),1)*out[:,1].reshape(len(out),1))-FORCE)**2
    #loss=(M*Psi_tt[:,0] +K*out[:,0:2]) ** 2
    #print("loss",loss.shape)
    loss=(M*A+out[:,1].reshape(len(out),1)*V+out[:,0].reshape(len(out),1)*D-F)**2
    # print("loss",loss.shape)
    mean_loss=torch.mean(loss)
    #print("mean_loss",mean_loss)
    return mean_loss,out


# =============================================================================
# We need to initialize the network
# =============================================================================
N = nn.Sequential(nn.Linear(1, 20,bias=True),
                  nn.Tanh(), 
                  nn.Linear(20,20,bias=True)
                  ,nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                   nn.Linear(20,20, bias=True),
                   nn.Tanh(),
                   nn.Linear(20,20, bias=True),
                   nn.Tanh(),
                   nn.Linear(20,20, bias=True),
                   nn.Tanh(),
                   nn.Linear(20,20, bias=True),
                   nn.Tanh(),
                  nn.Linear(20,20, bias=True),
                  nn.Tanh(),
                  nn.Linear(20,2, bias=True))
adam = torch.optim.Adam(N.parameters(), lr=0.001)
scheduler = StepLR(adam, step_size=25000, gamma=0.1)
# The batch size you want to use (how many points to use per iteration)




n_batch = 400

# x = ni * torch.rand(n_batch, 1)
# The maximum number of iterations to do
max_it =10000

for i in range(max_it):
    
    D,V,A = shuffle(DISP,VEL,ACC)
    
    #print(TX.shape)
    #X,Y,Z=X[0:n_batch,:],Y[0:n_batch,:],Z[0:n_batch,:]
   
    # Zero-out the gradient buffers
    adam.zero_grad()
    
    # Evaluate the loss
    l,pred_stiffness = loss(D,V,A,M,K,C)
    # Calculate the gradients
    l.backward()
    # Update the network
    adam.step()
    scheduler.step()
    # Print the iteration number
    if i % 100 == 99:
        print("epoch:",i+1,"loss",l.item())
print("stiffness",pred_stiffness)