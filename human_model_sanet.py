#!/usr/bin/env python3
from time import time
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import numpy as np
import pandas as pd
import os
from skimage import io, transform
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
from skimage.transform import resize

apply_batch_normalization=True
apply_dropout=False
dropout_prob = 0.2
gpu_num = 0

training_mode = 'state_action' # 'action' / 'state'
camera_mode = 'human' # 'human' / 'robot' 
color_channels = 3

use_cuda = torch.cuda.is_available()
device = torch.device(f"cuda:{gpu_num}" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

label_dict = {
    'athome': 0,
    'onconv': 1,
    'infront': 2,
    'atbin': 3,
    'unknown': 4
}

action_dict = {
    'detect': 0,
    'pick': 1,
    'inspect': 2,
    'placeonconv': 3,
    'placeinbin': 4,
    'noop':5
}

#----------------------------------------------------------------------------------------------------------------------------------#

class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = 0, 0
        self.bias = bias

        self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=3 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=3 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding='same',
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur = cur_state

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        input_conv = self.conv_input(input_tensor)
        h_conv = self.conv_h(h_cur)
        
        # combined_conv = input_conv + h_conv
        cc_r_i, cc_z_i, cc_h_i = torch.split(input_conv, self.hidden_dim, dim=1)
        cc_r_h, cc_z_h, cc_h_h = torch.split(h_conv, self.hidden_dim, dim=1)
        r = torch.sigmoid(cc_r_i + cc_r_h)
        z = torch.sigmoid(cc_z_i + cc_z_h)
        n = torch.tanh(cc_h_i + r *(cc_h_h))

        h_next = h_cur*z + n*(1-z)

        return h_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        output_height = int((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        output_width = int((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        return (
        torch.zeros(batch_size, self.hidden_dim, output_height, output_width, device=self.conv_input.weight.device),
        output_height, output_width)

#----------------------------------------------------------------------------------------------------------------------------------#

class ConvGRU(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
    Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          stride=self.stride[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h= self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            h,height, width = self.cell_list[i].init_hidden(batch_size, image_size)
            image_size = (height, width)
            init_states.append(h)
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

#----------------------------------------------------------------------------------------------------------------------------------#

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = 0, 0
        self.bias = bias

        self.conv_input = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=4 * self.hidden_dim,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_h = nn.Conv2d(in_channels=self.hidden_dim,
                                out_channels=4 * self.hidden_dim,
                                kernel_size=self.kernel_size,
                                padding='same',
                                bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        input_conv = self.conv_input(input_tensor)
        h_conv = self.conv_h(h_cur)
        combined_conv = input_conv + h_conv
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        output_height = int((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        output_width = int((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)

        return (
        torch.zeros(batch_size, self.hidden_dim, output_height, output_width, device=self.conv_input.weight.device),
        torch.zeros(batch_size, self.hidden_dim, output_height, output_width, device=self.conv_input.weight.device),
        output_height, output_width)

#----------------------------------------------------------------------------------------------------------------------------------#

class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
    Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, stride, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          stride=self.stride[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            h, c, height, width = self.cell_list[i].init_hidden(batch_size, image_size)
            image_size = (height, width)
            init_states.append((h, c))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

#----------------------------------------------------------------------------------------------------------------------------------#

class ActionDetect(nn.Module):
    def __init__(self, sequence_size=5):
        super(ActionDetect, self).__init__()
        self.sequence_size = sequence_size
        # TODO added depth image as 4th channel
        self.conv1 = nn.Conv2d(color_channels, 32, kernel_size=(2, 2), stride=(2, 2))
        # TODO regularization for kernel, torch.norm(model.layer.weight, p=2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(2, 2), stride=(2, 2))
        self.pooling1 = nn.AvgPool2d((2, 2), (1, 1))        
        self.convgru = ConvGRU(input_dim=16,
                                 hidden_dim=[20, 5],
                                 kernel_size=[(3, 3), (2, 2)],
                                 stride=[(2, 2), (3, 3)],
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)


        # self.fc = nn.Linear(7950,5)

    def forward(self, inputs, state_tensor=None):
        # TODO check the size of inputs, currently assume (batch_size, sequence_size, C, H, W)
        conv_outputs = []
        for i in range(self.sequence_size):
            output1 = F.leaky_relu(self.conv1(inputs[:, i, :, :, :]))
            output2 = F.leaky_relu(self.conv2(output1))
            conv_outputs.append(self.pooling1(output2))
        conv_outputs = torch.stack(conv_outputs)
        _, convgru_outputs = self.convgru(conv_outputs)
        convgru_h = convgru_outputs[0]
        convgru_h = torch.flatten(convgru_h, start_dim=1)

        return convgru_h

#----------------------------------------------------------------------------------------------------------------------------------#

class StateDetect(nn.Module):
    def __init__(self):
        super(StateDetect, self).__init__()
        self.conv1 = nn.Conv2d(color_channels, 32, kernel_size=(7, 7), stride=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(5, 5), stride=3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1)
        self.pool3 = nn.MaxPool2d(2, 2,1)
        
        self.batch_norm_2d_32 = nn.BatchNorm2d(32)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.leaky_relu(x) # Ehsan
        x = self.conv2(x)
        x = F.leaky_relu(x) # Ehsan
        x = self.conv3(x)
        x = F.leaky_relu(x) # Ehsan
        x = self.conv4(x)
        x = F.leaky_relu(x) # Ehsan
        x = self.conv5(x)
        x = F.leaky_relu(self.batch_norm_2d_32(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x) # Ehsan
        x = x.view(-1, 32* 9*3)   # Flatten layer
        return x

#----------------------------------------------------------------------------------------------------------------------------------#

class ActionClassifier(nn.Module):
    def __init__(self):
        super(ActionClassifier, self).__init__()

        self.fc1_1 = nn.Linear(2814, 128)
        self.fc1_2 = nn.Linear(128, 64)
        self.fc1_3 = nn.Linear(64, 32)
        self.fc1_out = nn.Linear(32, len(action_dict))  # for action
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.batch_norm_2d_32 = nn.BatchNorm2d(32)

        self.batch_norm_1d_32 = nn.BatchNorm1d(32)
        self.batch_norm_1d_64 = nn.BatchNorm1d(64)
        self.batch_norm_1d_128 = nn.BatchNorm1d(128)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.fc1_1.weight)
        nn.init.kaiming_normal_(self.fc1_2.weight)
        nn.init.kaiming_normal_(self.fc1_3.weight)
        nn.init.kaiming_normal_(self.fc1_out.weight)

    def forward(self, x_state, x_action):

        x = torch.cat((x_state, x_action), 1)
        # x = self.dropout(x) if apply_dropout else x
        x = self.fc1_1(x)
        x = F.leaky_relu(self.batch_norm_1d_128(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        # x = x #self.dropout(x) if apply_dropout else x
        x = self.fc1_2(x)
        x = F.leaky_relu(self.batch_norm_1d_64(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        # x = x #self.dropout(x) if apply_dropout else x
        x = self.fc1_3(x)
        x = F.leaky_relu(self.batch_norm_1d_32(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        
        action = self.fc1_out(x)  # for action
        
        # action = F.softmax(action, dim=1)

        return action

#----------------------------------------------------------------------------------------------------------------------------------#

class StateClassifier(nn.Module):
    def __init__(self):
        super(StateClassifier, self).__init__()

        self.fc1_1 = nn.Linear(32 * 9 * 3 ,128)
        self.fc1_2 = nn.Linear(128, 64)
        self.fc1_3 = nn.Linear(64, 32)
        self.fc1_out1 = nn.Linear(32, len(label_dict))  # for o_loc
        self.fc1_out2 = nn.Linear(32, len(label_dict))  # for ee_loc
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.batch_norm_1d_32 = nn.BatchNorm1d(32)
        self.batch_norm_1d_64 = nn.BatchNorm1d(64)
        self.batch_norm_1d_128 = nn.BatchNorm1d(128)

        self.init_weight()

    def init_weight(self):
        nn.init.kaiming_normal_(self.fc1_1.weight)
        nn.init.kaiming_normal_(self.fc1_2.weight)
        nn.init.kaiming_normal_(self.fc1_3.weight)
        nn.init.kaiming_normal_(self.fc1_out1.weight)
        nn.init.kaiming_normal_(self.fc1_out2.weight)

    def forward(self, x):
        
        x = self.dropout(x) if apply_dropout else x
        x = self.fc1_1(x)
        x = F.leaky_relu(self.batch_norm_1d_128(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        x = x # self.dropout(x) if apply_dropout else x
        x = self.fc1_2(x)
        x = F.leaky_relu(self.batch_norm_1d_64(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        x = x # self.dropout(x) if apply_dropout else x
        x = self.fc1_3(x)
        x = F.leaky_relu(self.batch_norm_1d_32(x)) if apply_batch_normalization and x.shape[0]>1 else F.leaky_relu(x)
        
        onion = self.fc1_out1(x)  # for o_loc
        eef = self.fc1_out2(x)  # for ee_loc
        
        # onion = F.softmax(onion, dim=1)
        # eef = F.softmax(eef, dim=1)

        return onion, eef

#----------------------------------------------------------------------------------------------------------------------------------#

class StateActionMain(nn.Module):
    def __init__(self):
        super(StateActionMain, self).__init__()

        # state 
        self.stateExpert = StateDetect()
        self.stateClassifier = StateClassifier()

        # action
        self.actionExpert = ActionDetect()
        self.actionClassifier = ActionClassifier()

    def forward(self, x_state,x_action):

        # state
        state_flatten = self.stateExpert(x_state)
        onion,eef = self.stateClassifier(state_flatten)


        # action
        action_flatten = self.actionExpert(x_action)
        action = self.actionClassifier(state_flatten,action_flatten)

        return onion,eef,action

#----------------------------------------------------------------------------------------------------------------------------------#

class StateActionDataset(Dataset):
    def __init__(self, state_csv_file, action_csv_file, rgb_dir, depth_dir, transform=None):
        self.state_labels = pd.read_csv(state_csv_file)
        self.action_csv_file = action_csv_file
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.transform = transform

        try:
            self.action_labels = pd.read_csv(action_csv_file)
        except FileNotFoundError:
            self.action_labels = self.create_action_csv()

    def __len__(self):
        return len(self.action_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # state
        file_name, o_loc, e_loc = self.state_labels.iloc[idx]

        # action
        action_data = self.action_labels.iloc[idx]
        action_images = action_data[:-1]
        action = action_data[-1]

        # state data process
        # depth_name = file_name.replace('frame','dframe')
        
        rgb_img_name = os.path.join(self.rgb_dir, file_name)
        # depth_img_name = os.path.join(self.depth_dir, depth_name)
        
        rgb_image = io.imread(rgb_img_name)
        # depth_image = io.imread(depth_img_name)
        
        # depth_image = np.reshape(depth_image,(depth_image.shape[0],depth_image.shape[1],-1))

        # state_data = np.concatenate((rgb_image,depth_image),axis=2)


        action_data = list()

        for img in action_images:
            # depth_img = img.replace('frame','dframe')
            rgb_img_name = os.path.join(self.rgb_dir, img)
            # depth_img_name = os.path.join(self.depth_dir, depth_img)

            rgb_image = io.imread(rgb_img_name)
            # depth_image = io.imread(depth_img_name)

            # depth_image = np.reshape(depth_image,(depth_image.shape[0],depth_image.shape[1],-1))

            # rgbd = np.concatenate((rgb_image,depth_image),axis=2)

            action_data.append(rgb_image)

        action_data = np.array(action_data)

        sample = {'state_image': rgb_image / 255 - 0.474, 'onion': np.array(label_dict[o_loc]), 'eef': np.array(label_dict[e_loc]),'action_image': action_data / 255, 'action': np.array(action_dict[action]), 'name': file_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def create_action_csv(self):
        index = 0
        action_labels = pd.DataFrame(columns=['image-5', 'image-4', 'image-3', 'image-2', 'image-1', 'action'])
        for i in range(len(self.labels)):
            if pd.isna(self.labels.iloc[i]['action']):
                continue
            action_labels.loc[index] = [self.labels.iloc[j]['image'] for j in range(i-4, i+1)] + [self.labels.iloc[i]['action']]
            index += 1
        action_labels.to_csv(self.action_csv_file, index=False)
        return action_labels

#----------------------------------------------------------------------------------------------------------------------------------#


class TestDataset(Dataset):
    def __init__(self, state_frame_list, action_frame_list, transform=None):
        
        self.state_frame_list = state_frame_list
        self.action_frame_list = action_frame_list
        self.transform = transform

    def __len__(self):
        return len(self.state_frame_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # state
        state_frame = self.state_frame_list[idx]

        # action
        # action_frames = self.action_frame_list[:]#[idx]

        # data process
        
        state_data = state_frame 

        action_data = np.array([state_frame]*5)

        sample = {'state_image': state_data / 255 - 0.474, 'action_image': action_data / 255,  'name': 'image'}

        if self.transform:
            sample = self.transform(sample)
        return sample

#----------------------------------------------------------------------------------------------------------------------------------#

class RescaleStateAction(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(480, 640)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        state_image = sample['state_image']

        h, w = state_image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        state_data = transform.resize(state_image, (new_h, new_w))

        rescaled_sample = {'state_image': state_data , 'onion': sample['onion'], 'eef': sample['eef'],'action_image': sample['action_image'], 'action': sample['action'], 'name': sample['name']}
        
        return rescaled_sample

#----------------------------------------------------------------------------------------------------------------------------------#

class ToTensorStateAction(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        state_image = sample['state_image']        
        action_image = sample['action_image']

        onion = sample['onion']
        eef = sample['eef']
        action = sample['action']
        name = sample['name']

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C x H x W
        state_image = torch.from_numpy(state_image.transpose((2, 0, 1)))
        action_image = torch.from_numpy(action_image.transpose((0, 3, 1, 2)))

        onion = torch.from_numpy(onion)
        eef = torch.from_numpy(eef)
        action = torch.from_numpy(action)

        tensor_sample = {'state_image': state_image,
                'action_image': action_image,
                'onion': onion,
                'eef': eef,
                'action': action,
                'name': name}

        return tensor_sample

#----------------------------------------------------------------------------------------------------------------------------------#

class ToTensorTest(object):

    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        state_image = sample['state_image']        
        action_image = sample['action_image']

        name = sample['name']

        # swap color axis because
        # numpy image: L x H x W x C
        # torch image: L x C x H x W

        # print('action_image shape: ', action_image.shape)
        state_image = torch.from_numpy(state_image.transpose((2, 0, 1)))
        action_image = torch.from_numpy(action_image.transpose((0, 3, 1, 2)))

        tensor_sample = {'state_image': state_image,
                'action_image': action_image,
                'name': name}

        return tensor_sample

#----------------------------------------------------------------------------------------------------------------------------------#

class RescaleTest(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(360, 640)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        state_image = sample['state_image']
        # action_image = sample['action_image']

        h, w = state_image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        state_data = resize(state_image, (new_h, new_w), anti_aliasing=True)
        action_data = np.array([state_data]*5)
        rescaled_sample = {'state_image': state_data , 'action_image': action_data, 'name': sample['name']}
        return rescaled_sample

#----------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------------------------------------------#

    