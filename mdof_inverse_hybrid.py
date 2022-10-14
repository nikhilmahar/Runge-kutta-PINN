# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:37:12 2022

@author: General
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import numpy as np
from math import sin
from scipy.linalg import eigh
from numpy.linalg import inv
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

time_step = 100
end_time = 10
class eulercell(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        
        super(eulercell, self).__init__()
        self.lin1 = nn.Linear(6, 6)
        # self.lin2 = nn.Linear(100, 100)
        # self.lin3 = nn.Linear(100, 100)
        # self.lin4 = nn.Linear(100, 100)
        # self.lin5 = nn.Linear(100, 100)
        # self.lin6 = nn.Linear(100, 6)
        # super().__init__()
        self.k1 = nn.Parameter(torch.rand(()))
        # nn.init.xavier_normal_(self.k1)
        self.k2 = nn.Parameter(torch.rand(()))
        # nn.init.xavier_normal_(self.k2)
        self.k3 = nn.Parameter(torch.rand(()))
        # self.K = nn.Parameter(torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]]))
        # nn.init.xavier_normal_(self.k3)
        # self.d = torch.nn.Parameter(torch.randn(()))
        self.initilize_weight()

    def forward(self, Y,act_acc):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # x = F.relu(self.lin1(x*self.k1))
        # print("k1",self.k1)
        # return self.lin2(x)
        
        
        F0 = torch.tensor(10.0)
        omega = torch.tensor(30)
        # k1 = k_pred[0]
       
        # k2= k_pred[1]
        # k3= k_pred[2]
        # print("k3",k3)
        m1 = torch.tensor(10.0)
        m2=torch.tensor(50.0)
        m3=torch.tensor(20.)
        dof = 3
       
        # time_step = 1
        # end_time = 10
        time=torch.linspace(0,time_step,end_time)
        # setup matrices
        # K = torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]])
        # K=self.K
        # K.requires_grad=True
        # print("k",K)
        M = torch.tensor([[m1,0,0],[0,m2,0],[0,0,m3]])
        I = torch.eye(dof)
       
        A = torch.zeros((2*dof,2*dof))
        B = torch.zeros((2*dof,2*dof))
        # Y = torch.zeros((2*dof,1))
        F1 = torch.zeros((2*dof,1))
       
        A[0:3,0:3] = M
        A[3:6,3:6] = I
       
        B[0:3,3:6] = torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]])
        B[3:6,0:3] = -I
     
       
        force=torch.zeros(len(time),1)
        X1=torch.zeros(len(time),1)
        X2=torch.zeros(len(time),1)
        X3=torch.zeros(len(time),1)
        A_inv = torch.inverse(A)
        acc1=torch.zeros(len(time),1)
        acc2=torch.zeros(len(time),1)
        acc3=torch.zeros(len(time),1)
        # print("Y",Y)
        # print("ainv",A_inv)
        for t in range(len(time)):
            # print("t",t)
            F1[1]=F0*torch.sin(omega*t)
            # print("F1",F1)
            # Y_new=Y+time_step*torch.matmul(A_inv,(F1-torch.matmul(B,Y.T))).T
            Y_new=self.lin1(Y+time_step*torch.matmul(A_inv,(F1-torch.matmul(B,Y.T))).T)
            # Y_new=F.tanh(self.lin2(Y_new))
            # Y_new=F.tanh(self.lin3(Y_new))
            # Y_new=F.tanh(self.lin4(Y_new))
            # Y_new=F.tanh(self.lin5(Y_new))
            # Y_new=self.lin6(Y_new)
            
            # return Y_new
            Y=Y_new
            Y1=Y_new
            Y1=Y1.T
            # print("YAFTER",Y)
            # Y_new=F.relu(self.lin1((Y.T+time_step*torch.matmul(A_inv,(F-torch.matmul(B,Y.T))))).T)
            # Y_new=Y+time_step*torch.matmul(A_inv,(F-torch.matmul(B,Y)))
            # Y=Y_new
            force[t,:]=F1[1]
            X1[t,:]=Y1[3]
            X2[t,:]=Y1[4]
            X3[t,:]=Y1[5]
            X_vec=torch.cat((X1[t,:],X2[t,:],X3[t,:])).reshape(dof,1)
            acc=torch.matmul(torch.inverse(M),(F1[0:3,:]-torch.matmul(torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]]),X_vec)))
            acc1[t,:]=acc[0]
            acc2[t,:]=acc[1]
            acc3[t,:]=acc[2]
        
        ACC=torch.hstack((acc1,acc2,acc3))
        # ACC=torch.tensor(ACC)
        # ACC=ACC.T
        DISP=torch.hstack((X1,X2,X3))
        
        l1=torch.matmul(M,act_acc) + torch.matmul(torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]]),Y1[3:6]) -F1[0:3]
        # print(l1)
        # return DISP, ACC
        pred_k=torch.tensor([[self.k1+self.k2,-self.k2,0],[-self.k2,self.k2+self.k3,-self.k3],[0,-self.k3,self.k3]])
        return Y,pred_k,DISP,ACC,l1
    
    def initilize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                
                nn.init.xavier_uniform_(m.weight)
        

    


def actual_response():
# =============================================================================
# data generation start using euler time integration scheme
# =============================================================================
# setup the parameters
    F0 = 10
    omega = 30
    k1 = 5.0
    k2= 2.5
    k3= 3
    m1 = 10.0
    m2=50.0
    m3=20.
    dof = 3
    
    time_step = 1
    end_time = 10
    
    # setup matrices
    K = np.array([[k1+k2,-k2,0],[-k2,k2+k3,-k3],[0,-k3,k3]])
    M = np.array([[m1,0,0],[0,m2,0],[0,0,m3]])
    I = np.identity(dof)
    
    A = np.zeros((2*dof,2*dof))
    B = np.zeros((2*dof,2*dof))
    Y = np.zeros((2*dof,1))
    F = np.zeros((2*dof,1))
    
    A[0:3,0:3] = M
    A[3:6,3:6] = I
    
    B[0:3,3:6] = K
    B[3:6,0:3] = -I
    
    # find natural frequencies and mode shapes
    # evals, phi = eigh(K,M)
    # frequencies = np.sqrt(evals)
    # print(frequencies)
    # print(phi)
    # mk=np.matmul(np.transpose(phi),K)
    # MK=np.matmul(mk,phi)
    # print(MK)
    
    A_inv = inv(A)
    force = []
    X1 = []
    X2 = []
    X3 = []
    # numerically integrate the EOMs
    for t in np.arange(0, end_time, time_step):
    	F[1] = F0 * sin(omega*t)
    	Y_new = Y + time_step * A_inv.dot( F - B.dot(Y) )
    	Y = Y_new
    	force.extend(F[1])
    	X1.extend(Y[3])
    	X2.extend(Y[4])
    	X3.extend(Y[5])
    f=np.zeros((dof,1))
    
    X_vec=np.zeros((3,1))
    X_vec1 = []
    X_vec2 = []
    X_vec3 = []
    acc1 = []
    acc2 = []
    acc3 = []
    acc=np.zeros((dof,1))
    for t1 in range(len(X1)):
        f[1]=force[t1]
        X_vec1=X1[t1]
        X_vec2=X2[t1]
        X_vec3=X3[t1]
        # print(X_vec1)
        X_vec=np.concatenate(([X_vec1],[X_vec2],[X_vec3])).reshape(dof,1)
        acc=np.matmul(inv(M),(f-np.matmul(K,X_vec)))
        acc1.extend(acc[0])
        acc2.extend(acc[1])
        acc3.extend(acc[2])
        
        act_acc=np.vstack(([acc1],[acc2],[acc3]))
        act_acc=torch.tensor(act_acc)
        act_acc=act_acc.T
        act_DISP=np.vstack(([X1],[X2],[X3]))
        act_DISP=torch.tensor(act_DISP)
        act_DISP=act_DISP.T
     
        
    
    return Y,act_DISP,act_acc


# Construct our model by instantiating the class defined above
model = eulercell()
act_STATES,act_DISP,act_ACC=actual_response()
act_STATES.T
act_STATES=torch.tensor(act_STATES.T).float()
dof=3
Y = torch.zeros((1,2*dof))  #initial state
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters (defined 
# with torch.nn.Parameter) which are members of the model.
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
scheduler = StepLR(optimizer, step_size=20000, gamma=0.1)
for t in range(50000):
    # Forward pass: Compute predicted y by passing x to the model
    
    # pred_STATES,PRED_K,PRED_DISP,PRED_ACC = model.forward(Y.float())
    pred_STATES,PRED_K,PRED_DISP,PRED_ACC,l1 = model.forward(Y.float(),act_ACC[9,:].float())
    # pred_STATES=pred_STATES.T

    # Compute and print loss
    loss1 = criterion(pred_STATES.float(), (act_STATES).float())
    loss2 = criterion(PRED_ACC[:,0].float(), (act_ACC[:,0]).float())
    loss3 = criterion(PRED_ACC[:,1].float(), (act_ACC[:,1]).float())
    loss4 = criterion(PRED_ACC[:,2].float(), (act_ACC[:,2]).float())
    loss5=criterion(l1.float(), torch.zeros(3,1).float())
    loss=loss1+loss2+loss3+loss4+loss5
    if t % 100 == 99:
        print("EPOCH",t,"loss1",loss1.item(),"loss2",loss2.item(),"loss3",loss3.item(),"loss4",loss4.item(),"LOSS", loss.item(),"Pred_K",PRED_K)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

# print(f'Result: {model.string()}')
# # plot results
# time = [round(t,5) for t in np.arange(0, end_time, time_step) ]
# time=torch.tensor(time)

# plt.plot(time,PRED_DISP[:,0].detach().numpy())
# plt.plot(time,PRED_DISP[:,1].detach().numpy())
# plt.plot(time,PRED_DISP[:,2].detach().numpy())

# plt.xlabel('time (s)')
# plt.ylabel('Pred displacement (m)')
# plt.title('Pred disp Response Curves')
# plt.legend(['X1', 'X2', 'X3'], loc='lower right')
# plt.show()
# plt.plot(time,act_DISP[:,0].detach().numpy())
# plt.plot(time,act_DISP[:,1].detach().numpy())
# plt.plot(time,act_DISP[:,2].detach().numpy())
# plt.xlabel('time (s)')
# plt.ylabel('act displacement ')
# plt.title('act disp Response Curves')
# plt.legend(['X1', 'X2', 'X3'], loc='lower right')
# plt.show()
