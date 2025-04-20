'''
Created on 20 apr 2025

@author: pasquale
'''
import torch.nn as nn
from torch import flatten

class leNet_arch(nn.Module):
	
	def __init__(self, numChannels, classes):
		super(leNet_arch, self).__init__()

		#numChan = 1 for greyscale or 3 for RGB
		self.conv1 = nn.Conv2d(in_channels=numChannels,
						 out_channels=50,
						 kernel_size = (5,5))
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))

		self.conv2 = nn.Conv2d(in_channels=50,
						 out_channels=100,
						 kernel_size = (10,10))
		self.relu2 = nn.ReLU()
		self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))

		self.fc1 = nn.Linear(in_features=1000, out_features=500)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(in_features=500, out_features=classes)
		self.logSoftmax = nn.LogSoftmax(dim=1)

	def forward(self, nex):
		
		#C1 S1 feature map
		nex = self.conv1(nex)
		nex = self.relu1(nex)
		nex = self.maxpool1(nex)
		
		#c2 s2 feature map
		nex = self.conv2(nex)
		nex = self.relu2(nex)
		nex = self.maxpool2(nex)
		
		#F3 Layer
		nex = flatten(nex, 1)
		nex = self.fc1(nex)
		nex = self.relu3(nex)
		
		#output layer and prediction probability
		nex = self.fc2(nex)
		return self.logSoftmax(nex)
