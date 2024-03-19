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

transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                          std=[0.2023, 0.1994, 0.2010])
                                     ])

dtd_dataset_train = datasets.DTD(root='DTD', split='train', download=True,transform=transform, partition=10)
dtd_dataset_val = datasets.DTD(root='DTD', split='val', download=True,transform=transform, partition=10)
dtd_dataset_test = datasets.DTD(root='DTD', split='test', download=True,transform=transform, partition=10)
dataset_train = ConcatDataset([dtd_dataset_train, dtd_dataset_val])
batch_size = 32
train_loader = data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = data.DataLoader(dtd_dataset_val, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(dtd_dataset_test, batch_size=batch_size, shuffle=True)

num_classes = 47
learning_rate = 0.01
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class dtdcl(nn.Module):
    def __init__(self, num_classes):
        super(dtdcl, self).__init__()
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

# Set Loss function with criterion
model = dtdcl(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

# Set optimizer with optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

total_step = len(train_loader)
# We use the pre-defined number of epochs to determine how many iterations to train the network on

writer = SummaryWriter('./logs')

for epoch in range(num_epochs):
	#Load in the data in batches using the train_loader object
    correct_pred = 0
    total_loss_per_epoch = 0.0
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images) # batch size x probabilites
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1)  # batchsize
        correct_pred += torch.eq(predictions, labels).sum().item()

        total_loss_per_epoch += loss.item() # loss -> [3.8] loss.item -> 3.8

    total_loss_per_epoch /= len(train_loader)
    accuracy = (correct_pred/len(dataset_train))*100
    writer.add_scalar('Loss/Train', total_loss_per_epoch, epoch)
    writer.add_scalar('Accuracy/Train', accuracy, epoch)
    print('Epoch [{}/{}], Loss: {:.4f} Accuracy {:.4f}'.format(epoch+1, num_epochs, total_loss_per_epoch,accuracy))

torch.save(model,'./DL_model.pt')      
torch.save(test_loader, './test_loader.pt')            
# loaded_model = torch.load('./DL_model.pt')

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = loaded_model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += torch.eq(predicted, labels).sum().item()
    
#     acc = 100 * correct / total
#     # writer.add_scalar('Loss/Test', test_loss, epoch)
#     writer.add_scalar('Accuracy/Test', acc)
#     print('Accuracy of the network on the {} train images: {} %'.format(50000, acc))
