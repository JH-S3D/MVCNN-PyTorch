import torch
import torch.nn as nn

class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            # b, 3, 224, 224
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),  
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=2)  # b, 16, 28, 28
        )

        self.fc_encoder = nn.Linear(16 * 28 * 28, 4096)
        self.fc_decoder = nn.Linear(4096, 16 * 28 * 28)

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ConvTranspose2d(16, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),  # b, 3, 223, 223
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        #print(x.size())
        x = self.encoder(x)
        #print(x.size())

        x = nn.Flatten(x)
        embedding = self.fc_encoder(x)

        embedding = self.fc_decoder(embedding)
        embedding = embedding.view(1, 16, 28, 28)

        x = self.decoder(embedding)
        #print(x.size())
        return x