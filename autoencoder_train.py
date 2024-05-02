import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast

import numpy as np
import time
import os

# Assuming MVCNN and MultiViewDataSet are defined in the imported module
from models.mvcnn import *
from custom_dataset import MultiViewDataSet

mse_loss = nn.MSELoss()

def train_model(model, dataloaders, device='cuda', num_epochs=25):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # Initialize the gradient scaler for AMP
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            num_batches = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # Assume inputs are already tensors
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():  # Using AMP for mixed precision
                        outputs, _ = model(inputs)
                        loss = mse_loss(outputs, inputs)

                    if phase == 'train':
                        scaler.scale(loss).backward()  # Scale the loss and call backward
                        scaler.step(optimizer)  # Optimizer step
                        scaler.update()  # Update the scaler
                    else:
                        running_loss += loss.item()

                num_batches += 1

            epoch_loss = running_loss / num_batches
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

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
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True, batch_size=1, num_workers=4)
                   for x in ['train', 'test']}

    model = mvcnn(pretrained=False)
    train_model(model, dataloaders, num_epochs=50, device='cuda')

if __name__ == '__main__':
    main()
