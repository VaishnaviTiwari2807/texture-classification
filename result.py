import torchvision.datasets as datasets
import torch.utils.data as data
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

class DEPNet(nn.Module):
    def __init__(self, num_classes):
        super(DEPNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),  # Adjust input size to match the output of the conv layers
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
       x = self.conv_layers(x)
       x = x.view(-1, 128 * 28 * 28)
       x = self.fc_layers(x)
       return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_model = torch.load('./DL_model_result.pt')
test_loader = torch.load('./test_loader_result.pt')
writer = SummaryWriter('./logs')
# images = images.to(device)
# labels = labels.to(device)
        

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += torch.eq(predicted, labels).sum().item()
    
    acc = 100 * correct / total
    #writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', acc)
    print('Accuracy of the network on the {} train images: {} %'.format(total,acc))