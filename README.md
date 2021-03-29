# GAN_Inversion and Checking the similar real images used in training time.

I used styleGANv1 trained with celebA-HQ. 

You can get CelebA_HQ dataset and pretrained StyleGAN in the following links.

## Download CelebA_HQ dataset
https://github.com/nperraud/download-celebA-HQ

## Pretrained StyleGAN with celebA-HQ in PyTorch
https://github.com/genforce/genforce

clone the repositoy(genforce) and put both genforce and this code in same directory. 


## This repository contains ....
First, this research is for analysing the GAN. Then it has two contents. 1. finding some real images similar to generated image. 2. GAN inversion(optimizing the latent vector to be real data(training dataset)

## What can we do with this code?
We can find the real data(real training dataset) which is similar to an image generated by generator.
And we can optimize the latent vector generating the generated image to be one of the real images which are similar to the generated image.
