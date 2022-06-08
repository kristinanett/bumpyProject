from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch


def compute_conv_dim(dim_size, kernel_size, padding, stride):
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)

class Net(nn.Module):
    def __init__(self, model_params):
        super(Net, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=model_params.channels_conv1,
                             out_channels=model_params.num_filters_conv1,
                             kernel_size=model_params.kernel_size_conv1,
                             stride=model_params.stride_conv1,
                             padding=model_params.padding_conv1)
        
        self.conv1_out_height = compute_conv_dim(model_params.img_height, model_params.kernel_size_conv1, model_params.padding_conv1, model_params.stride_conv1)
        self.conv1_out_width = compute_conv_dim(model_params.img_width, model_params.kernel_size_conv1, model_params.padding_conv1, model_params.stride_conv1)

        self.conv_2 = nn.Conv2d(in_channels=model_params.num_filters_conv1,
                             out_channels=model_params.num_filters_conv2,
                             kernel_size=model_params.kernel_size_conv2,
                             stride=model_params.stride_conv2,
                             padding=model_params.padding_conv2)
        
        self.conv2_out_height = compute_conv_dim(self.conv1_out_height, model_params.kernel_size_conv2, model_params.padding_conv2, model_params.stride_conv2)
        self.conv2_out_width = compute_conv_dim(self.conv1_out_width, model_params.kernel_size_conv2, model_params.padding_conv2, model_params.stride_conv2)

        #calculate nr of features that go into the fully connected layer
        #self.l1_in_features = .num_filters_conv1 * int(self.conv1_out_height) * int(self.conv1_out_width)   #two poolings mean height and width / 4
        self.l1_in_features = model_params.num_filters_conv2 * int(self.conv2_out_height) * int(self.conv2_out_width)

        self.l_1 = nn.Linear(in_features=self.l1_in_features, 
                            out_features=model_params.num_l1,
                            bias=True)

        self.l_2 = nn.Linear(in_features=model_params.num_l1, 
                            out_features=model_params.num_l2,
                            bias=True)

        #lstm layer
        self.lstm = nn.LSTM(input_size=model_params.input_size_lstm,
                         hidden_size=model_params.hidden_size_lstm,
                         num_layers=model_params.num_layers_lstm,
                         batch_first = True,
                         bidirectional=False)
        
        # Output layer
        self.l_out = nn.Linear(in_features=model_params.hidden_size_lstm,
                            out_features=model_params.num_lout,
                            bias=False)

        #dropout
        self.dropout1 = nn.Dropout(p=model_params.p_dropout_conv)
        self.dropout2 = nn.Dropout(p=model_params.p_dropout_conv)
        self.dropout3 = nn.Dropout(p=model_params.p_dropout_lin)
        self.dropout4 = nn.Dropout(p=model_params.p_dropout_lin)

        #batch normalization
        self.batchNorm_conv1 = nn.BatchNorm2d(model_params.num_filters_conv1)
        self.batchNorm_conv2 = nn.BatchNorm2d(model_params.num_filters_conv2)
        self.batchNorm_l1 = nn.BatchNorm1d(model_params.num_l1)
        self.batchNorm_l2 = nn.BatchNorm1d(model_params.num_l2)

    def forward(self, x):
        x_img = x[0]
        x_com = x[1]

        ################## IMG part ##############################

        #convolutional layer one
        #print(x_img.size()) #torch.Size([1, 3, 732, 1490])
        x_img = self.conv_1(x_img)
        x_img = F.relu(self.dropout1(x_img)) #torch.Size([1, 32, 364, 743])

        #convolutional layer two
        x_img = self.conv_2(x_img)
        x_img = self.batchNorm_conv2(F.relu(self.dropout2(x_img)))

        #2 fully connected layers
        x_img = x_img.view(-1, self.l1_in_features) #flatten
        x_img = F.relu(self.dropout3(self.l_1(x_img))) #torch.Size([1, 100])
        x_img = self.batchNorm_l2(F.relu(self.dropout4(self.l_2(x_img)))) #torch.Size((32, 64))

        ################## LSTM part ###################################

        #set image part output as the first hidden state to the LSTM
        h0 = x_img.reshape(1, x_com.size(0), x_img.size(-1)).requires_grad_() #1, 32, 64
        c0 = torch.zeros(1, x_com.size(0), x_img.size(-1)).requires_grad_() #x_com.size(0) is batch size

        h00 = torch.cat((h0, h0, h0, h0, h0, h0, h0, h0), 0) #for LSTM with 8 layers (8, 32, 64)
        c00 = torch.cat((c0, c0, c0, c0, c0, c0, c0, c0), 0)

        if torch.cuda.is_available():
            h00,c00 = h00.cuda() , c00.cuda()

        #print(x_com.size()) #torch.Size([1, 8, 2])
        x_com, (h_n, c_n) = self.lstm(x_com, (h00, c00)) #(h0.detach(), c0.detach()))

        #x55 = x1.view(-1, 100) #torch.Size([8, 100])

        #print(x1.size()) #torch.Size([1, 8, 100])
        x_out = self.l_out(x_com)
        #print(x1.size()) #torch.Size([1, 8, 1])

        return x_out
    
#net = Net()
#print(net)