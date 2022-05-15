import torch
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Normalize, ToTensor
from src.models.model import Net
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn


# Params for the network should be defined in this code instead (or with hydra)
# seq_dim = 8
# input_dim = 28
# hidden_dim = 100
# layer_dim = 1
# output_dim = 10

#create dataset and dataloader
dataset = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([Normalize(), ToTensor()]))
dataloader = DataLoader(dataset, batch_size=1)
dataloader_iter = iter(dataloader)

#A testrun for the raining code
model = Net()
model.train()
    
for i in range(1):
     x, y = next(dataloader_iter)

outputs = model(x)

#params
# num_epochs = 5
# learning_rate = 0.1
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(Net.parameters(), lr=learning_rate)

# iter = 0
# for epoch in range(num_epochs):
#     for i, (x, y) in enumerate(dataloader):
#         # Load images as a torch tensor with gradient accumulation abilities
#         x = x.requires_grad_()

#         # Clear gradients w.r.t. parameters
#         optimizer.zero_grad()

#         # Forward pass to get output/logits
#         outputs = model(images)

#         # Calculate Loss: mse loss
#         loss = criterion(outputs, labels)

#         # Getting gradients w.r.t. parameters
#         loss.backward()

#         # Updating parameters
#         optimizer.step()

#         iter += 1

#         if iter % 500 == 0:
#             # Calculate Accuracy         
#             correct = 0
#             total = 0
#             # Iterate through test dataset
#             for images, labels in test_loader:
#                 # Resize images
#                 images = images.view(-1, seq_dim, input_dim)

#                 # Forward pass only to get logits/output
#                 outputs = model(images)

#                 # Get predictions from the maximum value
#                 _, predicted = torch.max(outputs.data, 1)

#                 # Total number of labels
#                 total += labels.size(0)

#                 # Total correct predictions
#                 correct += (predicted == labels).sum()

#             accuracy = 100 * correct / total

#             # Print Loss
#             print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))