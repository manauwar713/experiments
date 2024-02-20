import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel ,self).__init__()
        self.fc1 = nn.Linear(12,500)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(500,100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(100,1)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x),dim=1)
        return x
        
        
        