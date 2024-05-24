import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),  # b, 64, 112, 112
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 56, 56
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # b, 256, 28, 28
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # b, 512, 14, 14
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # b, 1024, 7, 7
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True)
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024 * 7 * 7),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        embedding = self.fc_encoder(x)
        x = self.fc_decoder(embedding)
        x = x.view(-1, 1024, 7, 7)
        x = self.decoder(x)
        return x
    
    def get_embedded_vector(self, x):
        with torch.no_grad():
            x = self.encoder(x)
            x = self.flatten(x)
            embedding = self.fc_encoder(x)
        return embedding