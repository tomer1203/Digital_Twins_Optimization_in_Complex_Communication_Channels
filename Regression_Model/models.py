import torch as T
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_length,output_shape,model_output_capacity):
        super(Net, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)


        if model_output_capacity:
            self.hid1 = nn.Linear(input_size, hidden_size * 3)  # 8-(10-10)-1
            self.dropout1 = nn.Dropout(0.1)
            self.bn1 = nn.BatchNorm1d(hidden_size * 3)
            self.hid2 = nn.Linear(3 * hidden_size, hidden_size * 2)
            self.bn2 = nn.BatchNorm1d(hidden_size * 2)
            self.hid3 = nn.Linear(2 * hidden_size, hidden_size)
            self.bn3 = nn.BatchNorm1d(hidden_size)
            self.hid4 = nn.Linear(hidden_size, hidden_size//2)
            self.dropout2 = nn.Dropout(0.1)
            self.oupt = nn.Linear(hidden_size//2, 1)
        else:
            self.hid1 = nn.Linear(input_size, hidden_size * 6)  # 8-(10-10)-1
            self.dropout1 = nn.Dropout(0.1)
            self.bn1 = nn.BatchNorm1d(hidden_size * 6)
            self.hid2 = nn.Linear(6 * hidden_size, hidden_size * 4)
            self.bn2 = nn.BatchNorm1d(hidden_size * 4)
            self.hid3 = nn.Linear(4 * hidden_size, hidden_size * 3)
            self.bn3 = nn.BatchNorm1d(hidden_size * 3)
            self.hid4 = nn.Linear(3 * hidden_size, 8 * hidden_size)
            self.dropout2 = nn.Dropout(0.1)
            self.hid5 = nn.Linear(8 * hidden_size, 16 * hidden_size)
            self.oupt = nn.Linear(16*hidden_size, output_length * prod(output_shape))
            # self.hid1 = nn.Linear(input_size, hidden_size * 6)  # 8-(10-10)-1
            # self.dropout1 = nn.Dropout(0.1)
            # self.bn1 = nn.BatchNorm1d(hidden_size * 6)
            # self.hid2 = nn.Linear(6 * hidden_size, hidden_size * 8)
            # self.bn2 = nn.BatchNorm1d(hidden_size * 8)
            # self.hid3 = nn.Linear(8 * hidden_size, hidden_size * 12)
            # self.bn3 = nn.BatchNorm1d(hidden_size * 12)
            # self.hid4 = nn.Linear(12 * hidden_size, 24*hidden_size)
            # self.dropout2 = nn.Dropout(0.1)
            # self.oupt = nn.Linear(24*hidden_size, output_length*prod(output_shape))

        nn.init.xavier_uniform_(self.hid1.weight)
        nn.init.zeros_(self.hid1.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid2.weight)
        nn.init.zeros_(self.hid2.bias)
        nn.init.xavier_uniform_(self.hid3.weight)
        nn.init.zeros_(self.hid3.bias)
        nn.init.xavier_uniform_(self.oupt.weight)
        nn.init.zeros_(self.oupt.bias)

    def forward(self, x):

        z = T.relu(self.hid1(x**2))
        # if x.shape[0] != 1:  # batch_size==1
        z = self.bn1(z)
        # z = self.dropout1(z)
        z = T.relu(self.hid2(z))
        z = self.bn2(z)
        # z = self.bn2(z)
        z = T.relu(self.hid3(z))
        # z = self.bn3(z)
        z = T.relu(self.hid4(z))
        z = T.relu(self.hid5(z))
        # z = self.dropout2(z)
        z = self.oupt(z)  # no activation

        return z
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_features, mid_features, out_features):
        super(BasicBlock, self).__init__()
        self.fc1 = nn.Linear(in_features,mid_features)
        self.bn1 = nn.BatchNorm1d(mid_features)
        self.fc2 = nn.Linear(mid_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.shortcut = nn.Sequential()


    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks,input_size,hidden_size,output_length,output_shape):
        super(ResNet, self).__init__()
        # calculate the product of a list
        prod = lambda lst: reduce(lambda x, y: x * y, lst)
        self.fc_in= nn.Linear( input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.layer1 = self._make_layer(block, hidden_size, hidden_size, num_blocks[0])
        self.layer2 = self._make_layer(block, hidden_size, hidden_size, num_blocks[1])
        self.layer3 = self._make_layer(block, hidden_size, hidden_size, num_blocks[2])
        self.layer4 = self._make_layer(block, hidden_size, hidden_size, num_blocks[3])
        self.fc_out = nn.Linear(hidden_size, output_length*prod(output_shape))

    def _make_layer(self, block, in_features,out_features, num_blocks):

        layers = []
        layers.append(block(in_features,out_features,out_features))
        for stride in range(num_blocks-1):
            layers.append(block(out_features,out_features,out_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc_in(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc_out(out)
        return out

def ResNet18(input_size,hidden_size,output_length,output_shape):
    return ResNet(BasicBlock, [2, 2, 2, 2],input_size,hidden_size,output_length,output_shape)