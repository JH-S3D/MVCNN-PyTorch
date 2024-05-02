import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['MVCNN', 'mvcnn']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class MVCNN(nn.Module):

    def __init__(self, embedding_size=4096):
        super(MVCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        # Embedding layer to adjust to exactly embedding_size dimensions if needed
        self.fc_encoder = nn.Linear(256 * 6 * 6, embedding_size)

        self.sigmoid = nn.Sigmoid()

        # Decoder
        self.fc_decoder = nn.Linear(embedding_size, 256 * 6 * 6)
        self.decoder = nn.Sequential(
            # Upsampling + Convolution to progressively restore dimensions
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Increase size

            nn.ConvTranspose2d(256, 192, kernel_size=3, padding=1, output_padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Further increase size

            nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2, output_padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # Further increase size
            
            nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=1.1475),  # Further increase size
        )

    def forward(self, x):
        #batch_size, num_views, c, h, w = x.size()
        #x = x.view(batch_size * num_views, c, h, w)  # Reshape to combine batch and views

        # Encode views
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        embeddings = self.fc_encoder(x)
        embeddings = self.sigmoid(embeddings)

        # Decode views
        x = self.fc_decoder(embeddings)
        x = x.view(-1, 256, 6, 6)
        x = self.decoder(x)
        x = self.sigmoid(x)

        # Reshape back to separate views
        #print(x.size())
        #x = x.view(12, 3, 224, 224)  # Assuming output shape matches input

        # Pooling across views (example: max pooling)
        #output, _ = torch.max(x, 1)

        return x, embeddings


def mvcnn(pretrained=False, **kwargs):
    r"""MVCNN model architecture from the
    `"Multi-view Convolutional..." <hhttp://vis-www.cs.umass.edu/mvcnn/docs/su15mvcnn.pdf>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MVCNN(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model