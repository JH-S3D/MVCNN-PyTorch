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
        # Define encoder layers but do not encapsulate in nn.Sequential
        # if needing to capture output for indices
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, return_indices=True)
        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Linear(256 * 6 * 6, embedding_size)
        self.sigmoid = nn.Sigmoid()

        # Define decoder
        self.fc_decoder = nn.Linear(embedding_size, 256 * 6 * 6)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 384, kernel_size=3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(384, 192, kernel_size=3, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(192, 64, kernel_size=5, padding=2)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 3, kernel_size=11, stride=4, padding=2, output_padding=1)

    def forward(self, x):
        # Encoding
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x, indices1 = self.pool1(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x, indices2 = self.pool2(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x, indices3 = self.pool3(x)
        x = self.flatten(x)
        x = self.fc_encoder(x)
        embeddings = self.sigmoid(x)

        # Decoding
        x = self.fc_decoder(embeddings)
        x = x.view(-1, 256, 6, 6)
        x = self.unpool3(x, indices3)
        x = self.deconv5(x)
        x = nn.ReLU()(x)
        x = self.deconv4(x)
        x = nn.ReLU()(x)
        x = self.deconv3(x)
        x = nn.ReLU()(x)
        x = self.unpool2(x, indices2)
        x = self.deconv2(x)
        x = nn.ReLU()(x)
        x = self.unpool1(x, indices1)
        x = self.deconv1(x)
        x = nn.ReLU()(x)

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