import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, ratio_dict, compression_percentage, measurement_rate=0.25):
        super(Model, self).__init__()

        self.measurement_rate = measurement_rate
        self.ratio_dict = ratio_dict
        self.compression_percentage = compression_percentage
        self.fc1 = nn.Linear(ratio_dict[compression_percentage], 500)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.1)
        
        self.fc2 = nn.Linear(500, 1089)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.1)
        
        self.conv1 = nn.Conv2d(1, 32, 7, 1, padding=3)
        nn.init.normal_(self.conv1.weight, mean=0, std=0.1)
        
        self.conv2 = nn.Conv2d(32, 32, 3, 1, padding=1)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.1)
        
        self.conv3 = nn.Conv2d(32, 16, 3, 1, padding=1)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.1)

        self.conv4 = nn.Conv2d(16, 16, 3, 1, padding=1)
        nn.init.normal_(self.conv4.weight, mean=0, std=0.1)

        self.conv5 = nn.Conv2d(16, 8, 3, 1, padding=1)
        nn.init.normal_(self.conv5.weight, mean=0, std=0.1)

        self.conv6 = nn.Conv2d(8, 1, 3, 1, padding=1)
        nn.init.normal_(self.conv6.weight, mean=0, std=0.1)

    def forward(self, x):
        x = x.view(-1, self.ratio_dict[self.compression_percentage])
        # Sparse representation learning
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 33, 33)
        x = x.unsqueeze(1)
        
        # Convolutional layers for feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # Reconstruction layer
        x = self.conv6(x)

        return x
