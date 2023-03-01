import torch.nn as nn
import torch
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP, self).__init__()

        self.conv1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
    
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(1,32, kernel_size=3, stride=1,padding=0,
                padding_mode='zeros',bias=False),
                nn.ReLU())
        
        self.conv2 = nn.Sequential(
                nn.ConstantPad2d((2,2,2,2),0),
                nn.Conv2d(32, 32, kernel_size=3, stride=2,padding=0,
                bias=False),
                nn.ReLU())
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, 
                bias=False, padding_mode='zeros'),
                nn.ReLU())
        self.num_feature = 64*4*1
        self.fc1 = nn.Sequential(nn.Linear(self.num_feature,256))
        self.fc2 = nn.Sequential(nn.Linear(256,256))
        self.fc3 = nn.Sequential(nn.Linear(256, 1))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)    
        x = x.view(-1, self.num_feature )
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class DeepQNetwork2(nn.Module):
    def __init__(self):
        super(DeepQNetwork2, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(7, 224, groups=7, kernel_size=3, stride=1, padding=0,
                padding_mode='zeros',bias=False),
                nn.ReLU())
        
        self.conv2 = nn.Sequential(
                nn.ConstantPad2d((2,2,2,2),0),
                nn.Conv2d(224, 224, groups=7, kernel_size=3, stride=1,padding=0,
                bias=False),
                nn.ReLU())
        
        self.conv3 = nn.Sequential(
                nn.Conv2d(224, 448, groups=7, kernel_size=3, stride=2, 
                bias=False, padding_mode='zeros'),
                nn.ReLU())

        self.conv4 = nn.Sequential(
                nn.Conv2d(448, 448, groups=7, kernel_size=3, stride=2, 
                bias=False, padding_mode='zeros'),
                nn.ReLU())
        
        self.num_feature = 448*5*2
        self.fc1 = nn.Sequential(nn.Linear(self.num_feature,512))
        self.fc2 = nn.Sequential(nn.Linear(512,256))
        self.fc3 = nn.Sequential(nn.Linear(256, 40))
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        #x = F.pad(x,(0, 0, 1, 0), "constant", -1)
        x = F.pad(x,(1, 1, 0, 1), "constant", 0)
        x = self.conv1(x)
        #x1= x.to('cpu').detach().numpy().copy()
        #print(x1[0][33])
 
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.num_feature)
        #x1= x.to('cpu').detach().numpy().copy()
        #import numpy as np

        #np.savetxt('test.csv', x1[0])

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x