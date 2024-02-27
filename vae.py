import numpy as np
import torch
from torch import nn
from torchvision.models.resnet import resnet18

class VAE(nn.Module):

    def __init__(self, img_size, nb_channels, latent_img_size, z_dim, rec_loss="xent", beta=1, delta=1):
        '''
        '''
        super(VAE, self).__init__()

        self.img_size = img_size
        self.nb_channels = nb_channels
        self.latent_img_size = latent_img_size
        self.z_dim = z_dim
        self.beta = beta
        self.rec_loss = rec_loss
        self.delta = delta
        self.nb_conv = int(np.log2(img_size // latent_img_size))
        # the depth we will have at the end of the encoder given that a
        # convolution incease depth by 2 starting at 32 after the first
        self.max_depth_conv = 2 ** (4 + self.nb_conv)
        
        self.resnet = resnet18(pretrained=False)
        self.dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.dino_entry = nn.Sequential(
            nn.Conv2d(self.nb_channels, 384, kernel_size=7,
                stride=2, padding=3, bias=False)
        )
        self.resnet_entry = nn.Sequential(
            nn.Conv2d(self.nb_channels, 64, kernel_size=7,
                stride=2, padding=3, bias=False),
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool
        )
        self.resnet18_layer_list = [
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4 
        ]
        self.encoder_layers = [self.resnet_entry] 
        for i in range(1, self.nb_conv): 
            try:
                self.encoder_layers.append(self.resnet18_layer_list[i - 1])
            except IndexError: 
                depth_in = 2 ** (4 + i)
                depth_out = 2 ** (4 + i + 1)
                self.encoder_layers.append(nn.Sequential(
                    nn.Conv2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                    ))
        self.conv_encoder = nn.Sequential(
            *self.encoder_layers,
        )
        self.final_encoder = nn.Sequential(
            nn.Conv2d(self.max_depth_conv, self.z_dim * 2, kernel_size=1,
            stride=1, padding=0)
        )

        self.initial_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.max_depth_conv,
                kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.max_depth_conv),
            nn.ReLU()
        )
            
        nb_conv_dec = self.nb_conv

        self.decoder_layers = []
        for i in reversed(range(nb_conv_dec)):
            depth_in = 2 ** (4 + i + 1)
            depth_out = 2 ** (4 + i)
            if i == 0:
                depth_out = self.nb_channels
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                ))
            else:
                self.decoder_layers.append(nn.Sequential(
                    nn.ConvTranspose2d(depth_in, depth_out, 4, 2, 1),
                    nn.BatchNorm2d(depth_out),
                    nn.ReLU()
                ))
        self.conv_decoder = nn.Sequential(
            *self.decoder_layers
        )


    def encoder(self, x):
        x = self.conv_encoder(x)
        x = self.final_encoder(x)
        return x[:, :self.z_dim], x[:, self.z_dim:]

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(torch.mul(logvar, 0.5))
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decoder(self, z):
        z = self.initial_decoder(z)
        x = self.conv_decoder(z)
        x = nn.Sigmoid()(x)
        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        self.mu = mu
        print(mu.shape)
        self.logvar = logvar
        return self.decoder(z), (mu, logvar)

    def xent_continuous_ber(self, recon_x, x, pixelwise=False):
        ''' p(x_i|z_i) a continuous bernoulli '''
        eps = 1e-6
        def log_norm_const(x):
            # numerically stable computation
            x = torch.clamp(x, eps, 1 - eps)
            x = torch.where((x < 0.49) | (x > 0.51), x, 0.49 *
                    torch.ones_like(x))
            return torch.log((2 * self.tarctanh(1 - 2 * x)) /
                            (1 - 2 * x) + eps)
        if pixelwise:
            return (x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x))
        else:
            return torch.sum(x * torch.log(recon_x + eps) +
                            (1 - x) * torch.log(1 - recon_x + eps) +
                            log_norm_const(recon_x), dim=(1, 2, 3))

    def mean_from_lambda(self, l):
        ''' because the mean of a continuous bernoulli is not its lambda '''
        l = torch.clamp(l, 10e-6, 1 - 10e-6)
        l = torch.where((l < 0.49) | (l > 0.51), l, 0.49 *
            torch.ones_like(l))
        return l / (2 * l - 1) + 1 / (2 * self.tarctanh(1 - 2 * l))

    def kld(self):
        # NOTE -kld actually
        return 0.5 * torch.sum(
                1 + self.logvar - self.mu.pow(2) - self.logvar.exp(),
            dim=(1)
        )

    def loss_function(self, recon_x, x):
        rec_term = self.xent_continuous_ber(recon_x, x)
        rec_term = torch.mean(rec_term)

        kld = torch.mean(self.kld())

        L = (rec_term + self.beta * kld)

        loss = L

        loss_dict = {
            'loss': loss,
            'rec_term': rec_term,
            '-beta*kld': self.beta * kld
        }

        return loss, loss_dict

    def step(self, input_mb):
        recon_mb, _ = self.forward(input_mb)

        loss, loss_dict = self.loss_function(recon_mb, input_mb)

        recon_mb = self.mean_from_lambda(recon_mb)

        return loss, recon_mb, loss_dict

    def tarctanh(self, x):
        return 0.5 * torch.log((1+x)/(1-x))

        
