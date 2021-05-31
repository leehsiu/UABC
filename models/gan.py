import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    #Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
    #https://arxiv.org/abs/1511.06434
    def __init__(self,in_channels,ndf=64,ndf_max=512,n_layers=3):
        super(PatchDiscriminator, self).__init__()
        sequence = [nn.Conv2d(in_channels,ndf,7,1,3), nn.LeakyReLU(0.2, True)]
        in_channels = ndf
        for _ in range(n_layers):
            out_channels = min(in_channels*2,ndf_max)
            sequence += [
                nn.Conv2d(in_channels,out_channels,4,2,1,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            ]
            in_channels = out_channels
        sequence += [
            nn.Conv2d(in_channels,in_channels,3,1,1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(in_channels,1,3,1,1)] 

        self.model = nn.Sequential(*sequence)
        self.reset_parameters()
    
    def reset_parameters(self):
        for l in self.model.modules():
            if isinstance(l,nn.Conv2d):
                nn.init.kaiming_normal_(l.weight,a=1.0)
    def forward(self, input):
        return self.model(input)

class GANLoss(nn.Module):
    def __init__(self,mode='vanilla',real_label=1.0,fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.mode = mode
        if mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('GAN mode {} not implemented'.format(mode))

    def get_target_tensor(self, prediction, is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction,is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
            Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction,is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.mode == 'wgangp':
            if is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
            #loss = loss.clamp(-1,1)
        return loss
