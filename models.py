import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.nn.parameter import Parameter

class CellGRU(nn.Module):
    def __init__(self, in_features, out_features, units=100, prob=False):
        super(CellGRU, self).__init__()
        self.units = units
        self.prob = prob
        self.h = torch.zeros(1, units) # init hidden state
        self.c_t = self.h

        self.w_xh = Parameter(torch.rand(in_features, units))

        ### To calculate c tilda ###
        self.w_h_c = Parameter(torch.rand(units, units))

        ### To calculate gamma u ###
        self.w_h_u = Parameter(torch.rand(units, units)) 

        ### To calculate actual output ###
        self.w_h_y = Parameter(torch.rand(units, out_features))

        nn.init.xavier_uniform_(self.w_xh)
        nn.init.xavier_uniform_(self.w_h_c)
        nn.init.xavier_uniform_(self.w_h_u)
        nn.init.xavier_uniform_(self.w_h_y)

    def reset_hidden_state(self):
        self.c_t = torch.zeros(1, self.units)
        self.h   = torch.zeros(1, self.units)

    def forward(self, inputs):
        ### Since c_t and a_t is the same, no need to add c_t as input ###
        x_t = torch.matmul(inputs, self.w_xh)
        x = x_t + self.h
        c_tilda = torch.tanh(torch.matmul(x, self.w_h_c))
        gamma_u = torch.sigmoid(torch.matmul(x, self.w_h_u))

        self.c_t = gamma_u * c_tilda + (1 - gamma_u) * self.c_t
        self.h = self.c_t

        y = torch.matmul(self.h, self.w_h_y)
        if(self.prob):
            y = F.softmax(y)

        return y, self.h

class CellLSTM(nn.Module):
    def __init__(self, in_features, out_features, units=100, prob=False):
        super(CellLSTM, self).__init__()
        self.units = units
        self.prob = prob
        self.h = torch.zeros(1, units).type(torch.FloatTensor)
        self.c_t = torch.zeros(1, units).type(torch.FloatTensor)

        self.w_xh = Parameter(torch.rand(in_features, units), requires_grad=True)
 
        ### Initialize the weights of update-forget-output gates ###
        self.w_h_c = Parameter(torch.rand(units, units), requires_grad=True) # for c_tilda
        self.w_h_u = Parameter(torch.rand(units, units), requires_grad=True)
        self.w_h_f = Parameter(torch.rand(units, units), requires_grad=True)
        self.w_h_o = Parameter(torch.rand(units, units), requires_grad=True)
        # self.w_hy  = Parameter(torch.rand(units, out_features), requires_grad=True)

        ### Glorot initializer ###
        # nn.init.xavier_uniform_(self.w_xh)
        #nn.init.xavier_uniform_(self.w_h_u)
        #nn.init.xavier_uniform_(self.w_h_f)
        #nn.init.xavier_uniform_(self.w_h_o)
        # nn.init.xavier_uniform_(self.w_hy)

    def reset_hidden_state(self):
        self.c_t = torch.zeros(1, self.units).type(torch.FloatTensor)
        self.h   = torch.zeros(1, self.units).type(torch.FloatTensor)

    def forward(self, inputs):
        x = torch.matmul(inputs, self.w_xh)

        if(torch.isnan(x).any()):
            print("INPUTS : ", inputs)
            print("X_t : ", x_t)
            print("H : ", self.h)
            print("X_t + H : ", x)

        c_tilda = torch.tanh(torch.matmul(x, self.w_h_c))
        gamma_u = torch.sigmoid(torch.matmul(x, self.w_h_u))
        gamma_f = torch.sigmoid(torch.matmul(x, self.w_h_f))
        gamma_o = torch.sigmoid(torch.matmul(x, self.w_h_o))

        self.c_t = torch.mul(gamma_u, c_tilda) + torch.mul(gamma_f, self.c_t)
        self.h   = torch.mul(gamma_o,torch.tanh(self.c_t))

        return self.c_t, self.h
