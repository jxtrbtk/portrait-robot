# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ###############################################79############################
""" This script is a derivated work from Udacity project "Generating face"
The script build a GAN model, train it over the CelebA database (Udacity)
Instead of keeping the results over the test vectors to display some sets
The script save them on disk and generate a video at the end
Have fun !


"""
# #############79##############################################################
#                                      #
__author__ = "jxtrbtk"                 #
__contact__ = "bYhO-bOwA-dIcA"         #
__date__ = "gIvI-lOsI-kA"              # Wed Dec 26 10:52:44 2018
__email__ = "j.t[4t]free.fr"           #
__version__ = "1.0"                    #
#                                      #
# ##################################79#########################################


# necessary imports
#------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import cv2
import os
 
from os.path import isfile, join


# Parameters 
#------------------------------------------------------------------------------
#data loader
data_dir = "processed_celeba_small"
batch_size = 16
img_size = 5*32
#model
d_conv_dim = 96
g_conv_dim = 128
z_size = 256
# optimizer
lrd = 0.0002
lrg = 0.0002
beta1=0.5
beta2=0.999
# training
n_epochs = 50
loop_coeff = 0.0        #  0.000347
loop_coeff_add = 0.0    # -0.000000000888 
# video export
pathIn = 'data'
pathOut = 'video20.avi'
fps = 30
celeb_order = ["00","01","02","03","04","05","06","07","08","09","10","11"]
start_idx = 35789    
end_idx = start_idx + 8520 + 1

LOOP_TRAINING = True
LOOP_MOVIE = True


#--- debug --------------------------------------------------------------------
#pathIn = 'data2'
#pathOut = 'video9.avi'
#fps = 1
#celeb_order = ["00","01","02","03","04","05","06","07","08","09","10","11"]
#start_idx = 3
#end_idx = 8
#LOOP_TRAINING = True
#LOOP_MOVIE = True
#img_size = 1*32
#batch_size = 1024
##model
#d_conv_dim = 16
#g_conv_dim = 16
#z_size = 16
#n_epochs = 10
#--- debug --------------------------------------------------------------------



#### UDACITY #########################
#------------------------------------------------------------------------------
def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param img_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """                                     
    dtransform = transforms.Compose([ transforms.Resize(image_size),
                                       transforms.ToTensor()])
    dset = datasets.ImageFolder(data_dir, transform=dtransform)
    dloader  = torch.utils.data.DataLoader(dset, batch_size=batch_size, drop_last=True, shuffle=True)
    return dloader

def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    val_min, val_max = feature_range # explode tuple
    scale = val_max-val_min          # calc the new feature range scale
    center = (val_max+val_min)/2     # calc the new feature range center
    x = x-0.5                        # recenter the source (scale is already 1)  
    x = x*scale+center               # translate in the new range 
    return x

def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """
    # classname will be something like:
    # `Conv`, `BatchNorm2d`, `Linear`, etc.
    classname = m.__class__.__name__

    classname_init_list = ["Conv2d","ConvTranspose2d","Linear"]
    if classname in classname_init_list: 
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)

# # Model

# helper conv function
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
                                                               # 64x64 x 3
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)    # 32x32 x conv_dim
        self.conv2 = conv(conv_dim, conv_dim*2, 4)             # 16x16 x 2 conv_dim
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4)           #  8x 8 x 4 conv_dim
        self.conv4 = conv(conv_dim*4, conv_dim*4, 4)           #  4x 4 x 8 conv_dim
 
        self.fc1 = nn.Linear(conv_dim*4*int(img_size/16)*int(img_size/16), conv_dim*2)
        self.fc2 = nn.Linear(conv_dim*2, conv_dim)
        self.fc3 = nn.Linear(conv_dim, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.38)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        # 
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x) 
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x) 
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x) 
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x) 
        
        # flatten
        x = x.view(-1, self.conv_dim*4*int(img_size/16)*int(img_size/16))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x) 
        x = self.fc3(x)
        x = self.sigmoid(x)        
        
        return x


# ## Generator

# helper deconv function
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer =nn.ConvTranspose2d(in_channels, out_channels, 
                           kernel_size, stride, padding, bias=False)
    
    # append conv layer
    layers.append(conv_layer)

    if batch_norm:
        # append batchnorm layer
        layers.append(nn.BatchNorm2d(out_channels))
     
    # using Sequential container
    return nn.Sequential(*layers)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        self.conv_dim = conv_dim
        self.fc = nn.Linear(z_size, conv_dim*8*int(img_size/32)*int(img_size/32))
                                                               #  4x  4 x 8 conv_dim 
        self.deconv1 = deconv(conv_dim*8, conv_dim*4, 4)       #  8x  8 x 2 conv_dim
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)       # 16x 16 x conv_dim
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)         # 32x 32 x conv_dim
        self.deconv4 = deconv(conv_dim, conv_dim, 4)           # 64x 64 x conv_dim
        self.deconv5 = deconv(conv_dim, conv_dim, 4)           #128x128 x conv_dim
        self.deconv6 = deconv(conv_dim, conv_dim, 4)           #256x256 x conv_dim
        #self.deconv7 = deconv(conv_dim, int(conv_dim/2), 4)    #512x512 x conv_dim/2

        #self.conv1 = conv(int(conv_dim/2), conv_dim, 4)        #256x256 x conv_dim  
        self.conv2 = conv(conv_dim, 3, 4, batch_norm=False)    #128x128 x 3  
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.22)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = self.dropout(x)

        x = x.view(-1, self.conv_dim*8, int(img_size/32), int(img_size/32))
        
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)
        x = self.deconv3(x)
        x = self.relu(x)
        x = self.deconv4(x)
        x = self.relu(x)
        x = self.deconv5(x)
        x = self.relu(x)
        x = self.deconv6(x)
        x = self.relu(x)
        #x = self.deconv7(x)
        #x = self.relu(x)
        #x = self.conv1(x)
        #x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)
        
        return x

def build_network(d_conv_dim, g_conv_dim, z_size):
    # define discriminator and generator
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    # initialize model weights
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)
    
    return D, G
 
def calc_loss(D_out, labels, train_on_gpu):
    # move labels to GPU if needed     
    if train_on_gpu:
        labels = labels.cuda()
    criterion = nn.BCELoss()
    # calculate loss
    loss = criterion(D_out, labels)
    return loss

def real_loss(D_out, train_on_gpu, smooth):
    '''Calculates how close discriminator outputs are to being real.
       param, D_out: discriminator logits
       return: real loss'''
    batch_size = D_out.size(0)
    labels = (torch.ones(batch_size)*smooth).unsqueeze(1) # real labels = 1
    return calc_loss(D_out, labels, train_on_gpu)

def fake_loss(D_out, train_on_gpu):
    '''Calculates how close discriminator outputs are to being fake.
       param, D_out: discriminator logits
       return: fake loss'''
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size).unsqueeze(1) # fake labels = 0
    return calc_loss(D_out, labels, train_on_gpu)
    

def train(D, G, d_optimizer, g_optimizer, celeba_train_loader, train_on_gpu=True, n_epochs=3, z_size=100, print_every=50):
    '''Trains adversarial networks for some number of epochs
       param, D: the discriminator network
       param, G: the generator network
       param, n_epochs: number of epochs to train for
       param, print_every: when to print and record the models' losses
       return: D and G losses'''
    global loop_coeff
    global loop_coeff_add
    
    # move models to GPU
    if train_on_gpu:
        D.cuda()
        G.cuda()

    # keep track of loss and generated, "fake" samples
    loop_total_no = 0
    loop_score = 0.0

    samples_z = None
    img_generation = 0
    
    # Get some fixed data for sampling. These are images that are held
    # constant throughout training, and allow us to inspect the model's performance
    sample_size=12
    fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
    fixed_z = torch.from_numpy(fixed_z).float()
    # move z to GPU if available
    if train_on_gpu:
        fixed_z = fixed_z.cuda()

    if not os.path.exists(pathIn):
        os.mkdir(pathIn)
    for i in range(0,fixed_z.shape[0]):
        path = os.path.join(pathIn, "{0:02d}".format(i))
        if not os.path.exists(path):
            os.mkdir(path)

    # epoch training loop
    for epoch in range(n_epochs):

        # batch training loop
        for batch_i, (real_images, _) in enumerate(celeba_train_loader):

            batch_size = real_images.size(0)
            real_images = scale(real_images)

            # ===============================================
            #         YOUR CODE HERE: TRAIN THE NETWORKS
            # ===============================================

            if train_on_gpu:
                real_images = real_images.cuda()
            # 1. Train the discriminator on real and fake images
            D.train()
            G.eval()
            d_optimizer.zero_grad()
            # 1.1 real - pass real images, calculate loss assuming those are real images
            D_real = D(real_images)
            d_real_loss = real_loss(D_real, train_on_gpu, 1-np.random.normal()**2*0.1*(1-epoch/n_epochs))
            #d_real_loss = real_loss(D_real, train_on_gpu, 1.0)
            # 1.2 fake - pass fake images, calculate loss assuming those are fakes
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            fake_images = G(z)
            D_fake = D(fake_images)
            d_fake_loss = fake_loss(D_fake, train_on_gpu)
            # 1.3 overall loss and backprop
            d_loss = d_real_loss + d_fake_loss
            nn.utils.clip_grad_norm_(D.parameters(), 7)
            d_loss.backward()
            d_optimizer.step()
            
            # 2. Train the generator with an adversarial loss
            if ((epoch-1)/(n_epochs-1))**3 > np.random.rand():
                D.eval()
            else:
                D.train()
            #D.train() #weird, but seems not working in eval mode
            G.train()
            g_optimizer.zero_grad()
            # 1.1 generate images
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float()
            if train_on_gpu:
                z = z.cuda()
            gen_images = G(z)
            # 1.2 pass generated images to the discriminator to score them as real 
            D_gen = D(gen_images)
            g_loss = real_loss(D_gen, train_on_gpu, 1.0)
            nn.utils.clip_grad_norm_(G.parameters(), 7)
            g_loss.backward()
            g_optimizer.step()
            
            # Print some loss stats
            if batch_i % print_every == 0:
                # append discriminator loss and generator loss
#                 losses.append((d_loss.item(), g_loss.item()))
                # print discriminator and generator loss
                print('Epoch [{:3d}/{:3d} {:3.4f} {:6d}] | d_loss: {:6.4f} ({:6.4f}+{:6.4f}) | g_loss: {:6.4f}'.format(
                        epoch+1, n_epochs,loop_score,img_generation,d_loss.item(), d_real_loss.item(), d_fake_loss.item(), g_loss.item()))

            ## AFTER A NUMBER OF LOOPS ##    
            loop_score = loop_score + loop_coeff
            loop_coeff = loop_coeff + loop_coeff_add
            if loop_total_no % max(1,int(loop_score)) == 0:
                if img_generation >= start_idx and img_generation <= end_idx: 
                    G.eval() # for generating images
                    with torch.no_grad():
                        samples_z = G(fixed_z)
                        samples_z = samples_z.detach().cpu()
                        for i in range(0,samples_z.shape[0]):
                            img_path = os.path.join(pathIn, "{0:02d}".format(i), "{0:02d}_{1:08d}.png".format(i, img_generation))
                            save_image(samples_z[i], img_path, normalize=True)                        
                    G.train() # back to training mode
                img_generation = img_generation+1
            loop_total_no += 1
            #print('Epoch ]{:3d} | {:4.6f} | {:4.6f} ['.format(loop_total_no, loop_score,loop_coeff))

#            if img_generation >= end_idx: break
        
        ## End of Batch
        torch.save(G.state_dict(), "ModelG_"+str(epoch)+".pth")
        torch.save(D.state_dict(), "ModelD_"+str(epoch)+".pth")
        G.eval() # for generating images
        with torch.no_grad():
        	samples_z = G(fixed_z)
        	samples_z = samples_z.detach().cpu()
        	for i in range(0,samples_z.shape[0]):
        		img_path = os.path.join(pathIn, "{0:02d}_{1:08d}.png".format(i, img_generation))
        		save_image(samples_z[i], img_path, normalize=True)    
        G.train() # back to training mode
		
#        if img_generation >= end_idx: break
#------------------------------------------------------------------------------
#### UDACITY #########################


#https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
#------------------------------------------------------------------------------
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f)) and "02_" in f]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
#------------------------------------------------------------------------------
#https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
 
# https://stackoverflow.com/questions/47094930/creating-a-mosaic-of-thumbnails-with-opencv-3-and-numpy
#------------------------------------------------------------------------------
def get_images_mosaic(images, width=4):
    rows = []
    for i in range(0, len(images), width):
        rows.append(np.concatenate(images[i: i + width], axis=1))

    if len(rows) > 1:
        last_row = rows[-1]

        height = images[0].shape[0]
        last_row_width = last_row.shape[1]
        expected_width = rows[0].shape[1]

        if last_row_width < expected_width:
            filler = np.zeros((height, expected_width - last_row_width, 3))
            rows[-1] = np.concatenate((last_row, filler), axis=1)
        else:
            filler = None

    mosaic = np.concatenate(rows, axis=0)
    return mosaic 

def get_full_image(pathIn, celeb_order, index):
    image_parts = []
    for celeb in celeb_order:
        filename = os.path.join(pathIn,celeb,celeb + "_{:08d}.png".format(index))
        image_parts.append(filename)
    images = [cv2.imread(f) for f in image_parts]
    img = get_images_mosaic(images)
    return img
#------------------------------------------------------------------------------
# https://stackoverflow.com/questions/47094930/creating-a-mosaic-of-thumbnails-with-opencv-3-and-numpy



# main
#------------------------------------------------------------------------------
def main():

    if LOOP_TRAINING :

        # Call your function and get a dataloader
        celeba_train_loader = get_dataloader(batch_size, img_size, data_dir)
    
        #build model
        D, G = build_network(d_conv_dim, g_conv_dim, z_size)

        # Check for a GPU
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            print('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Training on GPU!')
    
        # Create optimizers for the discriminator and generator
        d_optimizer = optim.Adam(D.parameters(), lrd, [beta1, beta2])
        g_optimizer = optim.Adam(G.parameters(), lrg, [beta1, beta2])
    
        #train !
        train(D, G, d_optimizer, g_optimizer, celeba_train_loader, train_on_gpu, n_epochs=n_epochs, z_size=z_size, print_every=192)

    if LOOP_MOVIE :
	
        #generate video
        img = get_full_image(pathIn, celeb_order, start_idx)
        height, width, layers = img.shape
        size = (width,height)
        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for idx in range(start_idx, end_idx):
            img = get_full_image(pathIn, celeb_order, idx)
            out.write(img)
            if idx % 25 == 0 : print("frame {}".format(idx))
        out.release()


if __name__=="__main__":
    main()

#------------------------------------------------------------------------------
# main
    
