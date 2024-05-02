import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from models.img_autoencoder import ConvAutoencoder

from custom_dataset import MultiViewDataSet

data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
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

# defining the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining the model
convAE_model = ConvAutoencoder().to(device)

# defining the optimizer
optimizer = torch.optim.Adam(convAE_model.parameters(), lr= 0.001)

# defining the loss function
loss_function = torch.nn.MSELoss().to(device)

print(convAE_model)
print("____________________________________________________\n")

print("\nTraining the Convolutional AutoEncoder Model on Training Data...")

# Training of Model
losses = []
best_loss = float('inf')

for epoch in range(25): 
    epoch_loss = 0
    for X, y in dataloaders['train']:
        img = X[0].to(device)
        img = torch.autograd.Variable(img)
    
        recon = convAE_model(img)

        loss = loss_function(recon, img)
        
        # Backward Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss+= loss
        print('-', end= "", flush= True)

    epoch_loss = epoch_loss/len(dataloaders)
    losses.append(epoch_loss)
    if loss < best_loss:
        best_loss = loss
        print(f"New best loss {best_loss:.4f} at epoch {epoch+1}. Saving model...")
        torch.save(convAE_model.state_dict(), f'model_{epoch + 1}.pth')


    print("\nEpoch: {} | Loss: {:.4f}".format(epoch+1, epoch_loss))