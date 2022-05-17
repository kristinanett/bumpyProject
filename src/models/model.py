from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

num_l1 = 100
channels = 3
height = 732
width = 1490

#define convolutional layer parameters
num_filters_conv1 = 32
kernel_size_conv1 = 5
stride_conv1 = 2
padding_conv1 = 0


def compute_conv_dim(dim_size):
    return int((dim_size - kernel_size_conv1 + 2 * padding_conv1) / stride_conv1 + 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=channels,
                             out_channels=num_filters_conv1,
                             kernel_size=kernel_size_conv1,
                             stride=stride_conv1,
                             padding=padding_conv1)
        
        self.conv1_out_height = compute_conv_dim(height)
        self.conv1_out_width = compute_conv_dim(width)

        
        #calculate nr of features that go into the fully connected layer
        self.l1_in_features = num_filters_conv1 * int(self.conv1_out_height) * int(self.conv1_out_width)   #two poolings mean height and width / 4
        
        self.l_1 = nn.Linear(in_features=self.l1_in_features, 
                            out_features=num_l1,
                            bias=True)

        #lstm layer
        self.lstm = nn.LSTM(input_size=2,
                         hidden_size=100,
                         num_layers=1,
                         batch_first = True,
                         bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=100,
                            out_features=1,
                            bias=False)

    def forward(self, x):

        ################## img part ##############################

        #convolutional layer
        #print(x[0].size()) #torch.Size([1, 3, 732, 1490])
        x0 = self.conv_1(x[0]) 
        x0 = F.relu(x0)
        #print(x0.size()) #torch.Size([1, 32, 364, 743])

        #fully connected layer
        x0 = x0.view(-1, self.l1_in_features) #flatten
        x0 = F.relu(self.l_1(x0))
        #print(x0.size()) #torch.Size([1, 100])

        ################## LSTM part #############################

        #set image part output as the first hidden state to the LSTM
        h0 = x0.reshape(1, x[1].size(0), 100).requires_grad_()
        c0 = torch.zeros(1, x[1].size(0), 100).requires_grad_() #x[1].size(0) is batch size
        #print(x[1].size()) #torch.Size([1, 8, 2])
        x1,(h_n, c_n) = self.lstm(x[1], (h0.detach(), c0.detach()))

        #x55 = x1.view(-1, 100) #torch.Size([8, 100])

        #print(x1.size()) #torch.Size([1, 8, 100])
        x1 = self.l_out(x1)
        #print(x1.size()) #torch.Size([1, 8, 1])

        return x1
    
#net = Net()

# if torch.cuda.is_available():
#     print('##converting network to cuda-enabled')
#     net.cuda()

#print(net)