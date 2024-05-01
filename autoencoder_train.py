import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable

import numpy as np
import time
import os

# Assuming MVCNN and MultiViewDataSet are defined in the imported module
from models.mvcnn import *
from custom_dataset import MultiViewDataSet

mse_loss = nn.MSELoss()

def train_model(model, dataloaders, device='cuda', num_epochs=25):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer
    best_loss = float('inf')  # Initialize the best loss to infinity

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = np.stack(inputs, axis=1)
                #inputs = inputs[0]

                inputs = torch.from_numpy(inputs)

                inputs = inputs.cuda(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs, _ = model(inputs)  # Adjust based on your model's output
                    loss = mse_loss(outputs, inputs)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # Check if this is the best loss so far; save model if this is the case
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                print(f"New best loss {best_loss:.4f} at epoch {epoch+1}. Saving model...")
                torch.save(model.state_dict(), f'model_{epoch + 1}.pth')

def main():
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
    }

    data_dir = '/home/user/repo/modelnet40_images_new_12x'
    image_datasets = {x: MultiViewDataSet(root=data_dir, data_type=x, transform=data_transforms[x])
                      for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=4, num_workers=4)
                   for x in ['train', 'test']}

    model = mvcnn(pretrained=False)
    train_model(model, dataloaders, num_epochs=50, device='cuda')

if __name__ == '__main__':
    main()
