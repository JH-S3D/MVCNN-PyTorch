import torch
import torch.nn as nn

class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=1, padding=1),  # 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),  # b, 8, 2, 2

            torch.nn.Conv2d(16, 4, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        #self.fc_encoder = nn.Linear(16 * 222 *222, 4096)
        #self.fc_decoder = nn.Linear(4096, 16 * 222 *222)

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(4, 16, 3, stride=1, padding=2),  # b, 16, 10, 10
            torch.nn.ReLU(True),

            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 3, 3, stride=1, padding=2),  # b, 8, 3, 3
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        print(x.size())
        #x = nn.Flatten(x)
        #embedding = self.fc_encoder(x)

        #embedding = self.fc_decoder(embedding)
        #embedding = embedding.view(1, 16, 222, 222)

        x = self.decoder(x)
        return x