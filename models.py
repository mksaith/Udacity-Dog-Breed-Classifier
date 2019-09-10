import torchvision.models as models
import torch.nn as nn
import numpy as np

class Net_1(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 4, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 4, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=4608, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        x = x.view(-1, 4608)
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x

class Net_2(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 4, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size = 5, stride = 2, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=9216, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 9216)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    

class Net_3(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=15488, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 15488)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_4(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size = 5, stride = 3, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=12544, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 12544)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_5(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 64, kernel_size = 4, stride = 2, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=18496, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 18496)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_6(nn.Module):
    # same as Net_3 except for in_features
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=8192, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 8192)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_7(nn.Module):
    
    def __init__(self):
        super().__init__()
        # same as Net_3 & Net_6 except for in_features
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 2, bias = False)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=3200, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 3200)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_8(nn.Module):
    
    def __init__(self):
        super().__init__()
        # same as Net_3 & Net_6 except for in_features
        self.conv2d1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 2, padding = 2, bias = False)
        self.conv2d1_bn2d = nn.BatchNorm2d(128)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(128, 256, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.conv2d2_bn2d = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=6400, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 6400)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_9(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 5, stride = 1, bias = False)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 1, bias = False)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=4, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=10368, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 10368)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x

class Net_10(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding=2, bias = False)
        self.conv2d1_bn2d = nn.BatchNorm2d(128)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding=2, bias = False)
        self.conv2d2_bn2d = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=12544, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 12544)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x

class Net_11(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=10368, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 10368)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_12(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, bias=False)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, bias=False)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(in_features=4608, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 4608)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_13(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, bias=False)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 256, kernel_size = 3, stride = 1, bias=False)
        self.conv2d2_bn2d = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=9216, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 9216)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
    
class Net_14(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_relu1 = nn.ReLU() # LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, padding = 2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=8192, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 8192)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
    
    
class Net_15(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv2d0 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, bias = False)
        self.conv2d0_bn2d = nn.BatchNorm2d(64)
        
        self.conv2d1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 2, bias = False)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 2, bias = False)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=10368, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d0_bn2d( self.conv2d0(x) )
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.max_pool1( self.conv2d1_bn2d( self.conv2d1(x) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.max_pool2( self.conv2d2_bn2d( self.conv2d2(x) ) ) ) )
        
        x = x.view(-1, 10368)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_16(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.Conv2d0 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d0 = nn.BatchNorm2d(64)
        self.ReLU0 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout0 = nn.Dropout(p=0.1)
        
        self.Conv2d1 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d1 = nn.BatchNorm2d(64)
        self.MaxPool2d1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ReLU1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout1 = nn.Dropout(p=0.1)
        
        self.Conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d2 = nn.BatchNorm2d(128)
        self.ReLU2 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout2 = nn.Dropout(p=0.1)
        
        self.Conv2d3 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d3 = nn.BatchNorm2d(128)
        self.MaxPool2d3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ReLU3 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout3 = nn.Dropout(p=0.1)
        
        self.Conv2d4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d4 = nn.BatchNorm2d(256)
        self.ReLU4 = nn.LeakyReLU(negative_slope=0.1)
        self.Dropout4 = nn.Dropout(p=0.1)
        
        self.Conv2d5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d5 = nn.BatchNorm2d(256)
        self.MaxPool2d5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.ReLU5 = nn.LeakyReLU(negative_slope=0.1)
        self.Dropout5 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=4096, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.Dropout0( self.ReLU0( self.BatchNorm2d0( self.Conv2d0(x) ) ) )
        
        x = self.Dropout1( self.ReLU1( self.MaxPool2d1( self.BatchNorm2d1( self.Conv2d1(x) ) ) ) )
        
        x = self.Dropout2( self.ReLU2( self.BatchNorm2d2( self.Conv2d2(x) ) ) )
        
        x = self.Dropout3( self.ReLU3( self.MaxPool2d3( self.BatchNorm2d3( self.Conv2d3(x) ) ) ) )
        
        x = self.Dropout4( self.ReLU4( self.BatchNorm2d4( self.Conv2d4(x) ) ) )
        
        x = self.Dropout5( self.ReLU5( self.MaxPool2d5( self.BatchNorm2d5( self.Conv2d5(x) ) ) ) )
        
        x = x.view(-1, 4096)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_17(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.MaxPool2d1_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, bias=False) 
        self.MaxPool2d1_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.BatchNorm2d1 = nn.BatchNorm2d(64) 
        self.ReLU1 = nn.LeakyReLU(negative_slope=0.15)
        self.Dropout1 = nn.Dropout(p=0.15)
        
        self.MaxPool2d2_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d2 = nn.Conv2d(64, 256, kernel_size = 3, stride = 1, bias=False)
        self.MaxPool2d2_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.BatchNorm2d2 = nn.BatchNorm2d(256)
        self.ReLU2 = nn.LeakyReLU(negative_slope=0.15)
        self.Dropout2 = nn.Dropout(p=0.15)
        
        self.conv2d3 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, bias=False)
        self.BatchNorm2d3 = nn.BatchNorm2d(512)
        self.ReLU3 = nn.LeakyReLU(negative_slope=0.15)
        self.Dropout3 = nn.Dropout(p=0.15)
        
        self.fc1 = nn.Linear(in_features=18432, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.25)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.25)
        
    def forward(self, x):
        
        x = self.Dropout1( self.ReLU1( self.BatchNorm2d1( self.MaxPool2d1_2( self.conv2d1( self.MaxPool2d1_1(x) ) ) ) ) )
        
        x = self.Dropout2( self.ReLU2( self.BatchNorm2d2( self.MaxPool2d2_2( self.conv2d2( self.MaxPool2d2_1(x) ) ) ) ) )
        
        x = self.Dropout3( self.ReLU3( self.BatchNorm2d3( self.conv2d3(x) ) ) )
        
        x = x.view(-1, 18432)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_18(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.MaxPool2d1 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)    
        self.BatchNorm2d1 = nn.BatchNorm2d(64)
        self.ReLU1 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout1 = nn.Dropout(p=0.1)
        
        self.MaxPool2d2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.BatchNorm2d2 = nn.BatchNorm2d(128)
        self.ReLU2 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout2 = nn.Dropout(p=0.1)

        self.MaxPool2d3 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
        self.conv2d3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.BatchNorm2d3 = nn.BatchNorm2d(256)
        self.ReLU3 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout3 = nn.Dropout(p=0.1)
        
        self.MaxPool2d4 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 1)
        self.conv2d4 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.BatchNorm2d4 = nn.BatchNorm2d(256)
        self.ReLU4 = nn.ReLU() #nn.LeakyReLU(negative_slope=0.1)
        self.Dropout4 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=9216, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        x = self.Dropout1( self.ReLU1( self.BatchNorm2d1( self.conv2d1( self.MaxPool2d1(x) ) ) ) )
        
        x = self.Dropout2( self.ReLU2( self.BatchNorm2d2( self.conv2d2( self.MaxPool2d2(x) ) ) ) )
        
        x = self.Dropout3( self.ReLU3( self.BatchNorm2d3( self.conv2d3( self.MaxPool2d3(x) ) ) ) )
        
        x = self.Dropout4( self.ReLU4( self.BatchNorm2d4( self.conv2d4( self.MaxPool2d4(x) ) ) ) )
        
        x = x.view(-1, 9216)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_19(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.MaxPool2d1_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1) 
        self.MaxPool2d1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.BatchNorm2d1 = nn.BatchNorm2d(64) 
        self.ReLU1 = nn.LeakyReLU(negative_slope=0.1)
        self.Dropout1 = nn.Dropout(p=0.1)
        
        self.MaxPool2d2_1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.MaxPool2d2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.BatchNorm2d2 = nn.BatchNorm2d(128)
        self.ReLU2 = nn.LeakyReLU(negative_slope=0.1)
        self.Dropout2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=6272, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.Dropout1( self.ReLU1( self.BatchNorm2d1( self.MaxPool2d1_2( self.conv2d1( self.MaxPool2d1_1(x) ) ) ) ) )
        
        x = self.Dropout2( self.ReLU2( self.BatchNorm2d2( self.MaxPool2d2_2( self.conv2d2( self.MaxPool2d2_1(x) ) ) ) ) )
        
        x = x.view(-1, 6272)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_20(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.max_pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, bias=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=2048, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.conv2d1_bn2d( self.max_pool1( self.conv2d1( self.max_pool0(x) ) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.conv2d2_bn2d( self.max_pool2( self.conv2d2( x) ) ) ) ) 
        
        x = x.view(-1, 2048)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
class Net_21(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.max_pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2d_drouput0 = nn.Dropout(p=0.01)
        self.conv2d1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, bias=False)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d1_bn2d = nn.BatchNorm2d(64)
        self.conv2d_leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput1 = nn.Dropout(p=0.1)
        
        self.conv2d2 = nn.Conv2d(64, 128, kernel_size = 5, stride = 2, bias=False)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d2_bn2d = nn.BatchNorm2d(128)
        self.conv2d_leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2d_drouput2 = nn.Dropout(p=0.1)
        
        self.fc1 = nn.Linear(in_features=2048, out_features=1024, bias=False)
        self.fc1_bn1d = nn.BatchNorm1d(1024)
        self.fc1_relu = nn.ReLU()
        self.fc1_drouput = nn.Dropout(p=0.15)
        
        self.fc2 = nn.Linear(in_features=1024, out_features=133, bias=False)
        self.fc2_bn1d = nn.BatchNorm1d(133)
        self.fc2_drouput = nn.Dropout(p=0.15)
        
    def forward(self, x):
        
        x = self.conv2d_drouput1( self.conv2d_leaky_relu1( self.conv2d1_bn2d( self.max_pool1( self.conv2d1(self.conv2d_drouput0( self.max_pool0(x) ) ) ) ) ) )
        
        x = self.conv2d_drouput2( self.conv2d_leaky_relu2( self.conv2d2_bn2d( self.max_pool2( self.conv2d2( x) ) ) ) ) 
        
        x = x.view(-1, 2048)
        
        x = self.fc1_drouput(self.fc1_relu( self.fc1_bn1d( self.fc1( x ) ) ) )
        
        x = self.fc2_drouput(self.fc2_bn1d( self.fc2( x ) ) )
        
        return x
    
def weights_init(m):
    '''Takes in a module and initializes all linear layers with weight
       values taken from a normal distribution.'''
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def get_model(mod_idx):
    
    model=None
    augment=True
    image_size=None
    if mod_idx==1:
        model = Net_1()
        image_size=250
    elif mod_idx==2:
        model = Net_2()
        image_size=250
    elif mod_idx==3:
        model = Net_3()
        image_size=200
    elif mod_idx==4:
        model = Net_4()
        image_size=200
    elif mod_idx==5:
        model = Net_5()
        image_size=150
    elif mod_idx==6:
        model = Net_6()
        image_size=150
    elif mod_idx==7:
        model = Net_7()
        image_size=100
    elif mod_idx==8:
        model = Net_8()
        image_size=100
    elif mod_idx==9:
        model = Net_9()
        image_size=60
    elif mod_idx==10:
        model = Net_10()
        image_size=60
    elif mod_idx==11:
        model = Net_11()
        image_size=40
    elif mod_idx==12:
        model = Net_12()
        image_size=40
    elif mod_idx==13:
        model = Net_13()
        image_size=40
    elif mod_idx==14:
        model = Net_14()
        image_size=150
    elif mod_idx==15:
        model = Net_13()
        image_size=40
        augment=False
    elif mod_idx==16:
        model = Net_16()
        image_size=80
    elif mod_idx==17:
        model = Net_17()
        image_size=150
    elif mod_idx==18:
        model = Net_18()
        image_size=80
    elif mod_idx==19:
        model = Net_19()
        image_size=150
    elif mod_idx==20:
        model = Net_20()
        image_size=200
    elif mod_idx==21:
        model = Net_21()
        image_size=200
     
    model.apply(weights_init)
    
    return model, image_size, augment