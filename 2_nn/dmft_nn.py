import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import sys
#====================================================================#
parser = argparse.ArgumentParser()
parser.add_argument("-D", help = "half bandwidth", type = float, default = 1)
parser.add_argument("-T", help = "temperature", type = float)
parser.add_argument("-lr", help = "learning rate", type = float, default = 1e-3)
parser.add_argument("-reg", help = "Regularization", type = float, default = 1e-5)
args = parser.parse_args()
D = args.D
T = args.T
learning_rate = args.lr
reg = args.reg
h_node = 200
out_node = 2
#====================================================================#


if T == 0.001:
    add_dir = '_0.001'
    U_c1, U_c2 = 2.2, 2.59
elif T == 0.01:
    add_dir = ''
    U_c1, U_c2 = 2.2, 2.37
else:
    print('give temperature')
    sys.exit(1)

w_len = len(np.loadtxt(f'.{directory}/1to4{add_dir}/Bethe-{2.00:.2f}_solution.dat', \
                       unpack = True, dtype = 'complex128')[0])
directory = ''
U_i1, U_f2 = 100, 401
U_f1, U_i2 = int(U_c1*100), int(U_c2*100)

U = np.array([0.01*i for i in range(U_i1+50, U_f2-50)])
U1 = np.array([0.01*i for i in range(U_i1, U_f1)])
U2 = np.array([0.01*i for i in range(U_i2, U_f2)])
x = np.zeros((len(U1)+len(U2), w_len*2), dtype = 'float64')
x_test = np.zeros((len(U)*2, w_len*2), dtype = 'float64')

# train data
for i, u in enumerate(U1):
    w, A_w, G_w, S_w = np.loadtxt(f'.{directory}/1to4{add_dir}}/Bethe-{u:.2f}_solution.dat', \
                                    unpack = True, dtype = 'complex128')
    x[i][:w_len], x[i][w_len:] = ((D/2)**2*G_w).real, ((D/2)**2*G_w).imag
for i, u in enumerate(U2):
    w, A_w, G_w, S_w = np.loadtxt(f'.{directory}/4to1{add_dir}/Bethe-{u:.2f}_solution.dat', \
                                    unpack = True, dtype = 'complex128')
    x[i+len(U1)][:w_len], x[i+len(U1)][w_len:] = ((D/2)**2*G_w).real, ((D/2)**2*G_w).imag
x = torch.FloatTensor(x)
size = len(x[0])

# test data
for i, u in enumerate(U):
    w, A_w, G_w, S_w = np.loadtxt(f'.{directory}/1to4{add_dir}}/Bethe-{u:.2f}_solution.dat', \
                                    unpack = True, dtype = 'complex128')
    x[i][:w_len], x[i][w_len:] = ((D/2)**2*G_w).real, ((D/2)**2*G_w).imag
for i, u in enumerate(U):
    w, A_w, G_w, S_w = np.loadtxt(f'.{directory}/4to1{add_dir}/Bethe-{u:.2f}_solution.dat', \
                                    unpack = True, dtype = 'complex128')
    x[i+len(U)][:w_len], x[i+len(U)][w_len:] = ((D/2)**2*G_w).real, ((D/2)**2*G_w).imag
x_test = torch.FloatTensor(x_test)

# Metal = 1, Insulator = 0
y_temp1 = np.array(U1 <= U_c1)
y_temp2 = np.array(U2 < U_c2)
y_temp = np.concatenate([y_temp1, y_temp2])
y = np.stack([y_temp, ~y_temp], axis = 1)
y = torch.FloatTensor(y)

y_temp1 = np.array(U <= U_c1)
y_temp2 = np.array(U < U_c2)
y_temp = np.concatenate([y_temp1, y_temp2])
y_test = np.stack([y_temp, ~y_temp], axis = 1)
y_test = torch.FloatTensor(y_test)
U = np.tile(U, 2)

def divide(x_, y_):
    mask = np.random.rand(len(x_)) < 0.75  # train : val = 3 : 1
    x_train, x_val = x_[mask], x_[~mask]
    y_train, y_val = y_[mask], y_[~mask]
    return x_train, x_val, y_train, y_val

x_train, x_val, y_train, y_val = divide(x, y)

class Net(nn.Module):
    def __init__(self, node = None, activate = None):
        super(Net, self).__init__()
        self.node = node
        self.W1 = nn.Linear(size, self.node, bias=False)
        self.b1 = nn.Linear(self.node, 1, bias=False)
        self.W2 = nn.Linear(self.node, out_node, bias=False)
        self.b2 = nn.Linear(out_node, 1, bias=False)
        self.activate = activate()
        self.sig = nn.Sigmoid()

        if (activate==nn.Sigmoid or activate==nn.Tanh):
            nn.init.xavier_normal_(self.W1.weight)
            nn.init.xavier_normal_(self.W2.weight)
        else:
            nn.init.kaiming_normal_(self.W1.weight)
            nn.init.kaiming_normal_(self.W2.weight)
        self.b1.weight.data.fill_(0)
        self.b2.weight.data.fill_(0)

    def forward(self, x):
        return self.sig(self.W2(self.activate(self.W1(x)+self.b1.weight.data))+self.b2.weight.data)

    def loss1(self, output, y, reg): # regularization l1
        regular = reg*(torch.norm(self.W1.weight.data, p=1)+torch.norm(self.W2.weight.data, p=1))
        return F.binary_cross_entropy(output, y) + regular # cross entropy loss
    def loss2(self, output, y, reg): # regularization l2
        regular = reg*(torch.norm(self.W1.weight.data)**2+torch.norm(self.W2.weight.data)**2)
        return F.binary_cross_entropy(output, y) + regular # cross entropy loss

    def accuracy(self, output, y):
        return np.average((output>=0.5)==y)

func = nn.ReLU # activation function
model = Net(node = h_node, activate = func)

# Begin training and validation
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
n_epochs = 2000
train_loss, val_loss, test_acc = np.ones(n_epochs+1), np.ones(n_epochs+1), np.ones(n_epochs+1)
for epoch in range (n_epochs+1):
    loss = model.loss2(model.forward(x_train), y_train, reg)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    valid_loss = model.loss2(model.forward(x_val), y_val, reg).item()
    t_acc = model.accuracy(model.forward(x_test), y_test)
    train_loss[epoch], val_loss[epoch], test_acc[epoch] = loss.item(), valid_loss, t_acc

    if epoch%int(n_epochs/20) == 0:
        print('epoch = %d, training loss = %.8f, accuracy = %.8f' \
                %(epoch, loss.item(), t_acc))
    if epoch > 50:
        if valid_loss > val_loss[epoch-50]: # prevent overfitting
            print('epoch = %d, training loss = %.8f, accuracy = %.8f' \
                  %(epoch, loss.item(), model.accuracy(model.forward(x_test), y_test)))
            np.savetxt(f'./test_result.dat', np.array([U, model.forward(x_test).data[:,0], y_test[:,0], header = f"U   train_output   actual_output"))
            np.savetxt(f'./weight1_result.dat', np.array(model.W1.weight.data))
            np.savetxt(f'./weight2_result.dat', np.array(model.W2.weight.data))

            fig, ax = plt.subplots(1, 2, figsize = (10, 5))
            im1 = ax[0].imshow(np.array(model.W1.weight.data), aspect=size/h_node*0.5, cmap='bwr')
            im2 = ax[1].imshow(np.array(model.W2.weight.data), aspect=h_node/15, cmap='bwr')
            cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
            cd = fig.colorbar(im1, cax = cbar_ax)
            plt.savefig('weight_data.png')
            """ # plot
            fig, ax = plt.subplots(1,2, figsize=(12, 4))
            ax[0].plot(U, model.forward(x_test).data[:,0], 'b.', label = 'trained')
            ax[0].plot(U, y_test[:,0], 'r.', ms = 2, label = 'actual')
            ax[0].legend()
            ax[1].plot(np.arange(len(train_loss)), train_loss/np.max(val_loss), '-', label='train loss')
            ax[1].plot(np.arange(len(val_loss)), val_loss/np.max(val_loss), '-', label='validation')
            ax[1].plot(np.arange(len(test_acc)), test_acc, '-', label='test acc')
            ax[1].set_xlim(0, epoch)
            ax[1].legend()
            plt.show()
            fig, ax = plt.subplots(1, 2, figsize = (10, 5))
            im1 = ax[0].imshow(np.array(model.W1.weight.data), aspect=size/h_node*0.5, cmap='bwr')
            im2 = ax[1].imshow(np.array(model.W2.weight.data), aspect=h_node/15, cmap='bwr')
            cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
            cd = fig.colorbar(im1, cax = cbar_ax)
            plt.show()
            """
            break
