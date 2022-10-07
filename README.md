# CS 4641 Project: Artificial Neural Intelligence Machine Entity

by Abhay Sheshadri, Sainitin Daverpally, Nick Chapman, Abdul Sayed, Kevin Xiao

### Introduction/Background: 
Our idea is to create an AI model that colorizes and upscales a low-resolution black-and-white anime portrait image.  The people have been starved of colored content for their black and white manga pictures. The only way to see a manga page colored is either to color it yourself or to wait for the anime adaptation/anime release of that chapter. However, a vast majority of manga readers only have beginner/no artistic skills and waiting for the anime adaptation requires long waits. Especially when there are low quality scans of various manga on the internet, there is great demand to both color and upscale them for western consumer use.

### Problem definition:
Our problem is an image-to-image task.  There are several architectures that were previously used for this task such as CycleGan, PatchGAN, Pix2Pix.  These methods involve having to simultaneously learn the multiple distributions of images and the mappings between them.  We instead use methods that leverage a model which already has learned the distribution of images, which will allow us to produce high-quality images instead of blurry or discolored one. 

All of this team’s members share a similar sentiment, that low quality scans of great works be brought to life for many to enjoy!

### Methods:
Using our dataset of anime portraits, we can generate ground-truth image pairs for our task by downsampling the images and making them greyscale.   We then plan to use generative models to produce high resolution colored images from the input.  Instead of sampling from the unconditional distribution of anime portraits, we plan to condition the model on a particular low-resolution black and white anime image.  We will first train a generative adversarial network such as a StyleGAN3 model on the high-quality anime faces and then create another auxilliary model similar to what is proposed in Pix2Style2Pix [1] or PULSE [3] to invert the low-quality images into the StyleGAN latent space.  We can optionally use methods such as EditGAN [2] to further edit the output images.  We can then sample possible high-quality outputs that correspond to the low-resolution input.  We will implement the aforementioned models with the PyTorch and NumPy frameworks; we may also use MatPlotLib for data visualization.

### Potential results and Discussion:
At a basic level, our algorithm aims to return a colored, upscaled image starting from a low resolution, grayscale base. We will measure the accuracy of our method using both mean squared losses between the target and predicted image and perceptual losses as proposed in [4]. Since our goal is to produce several possible high-quality images for a given low-quality image, we need our model to be expressive.  We will measure the expressiveness of our generative model using metrics such as Frechet Inception Distance. 

### Dataset:

![alt text](https://i.ibb.co/b1q68w1/lol.png)


We intend to achieve this by using the anime-faces dataset by Gwern created from scraping various other online resources.

Samples from the dataset
Some metadata on the images
Image_Size: 256x256 pixels
Number Of Images: 226,037

### GANTT
https://onedrive.live.com/view.aspx?resid=FDDA84B319FFA688!83348&ithint=file%2cxlsx&authkey=!AHuEvXf-Bykv6ng

### Video: 


References:
[1] Richardson, E., Alaluf, Y., Patashnik, O., Nitzan, Y., Azar, Y., Shapiro, S., & Cohen-Or, D.. (2020). Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation.
[2] Ling, H., Kreis, K., Li, D., Kim, S., Torralba, A., & Fidler, S.. (2021). EditGAN: High-Precision Semantic Image Editing.
[3] Menon, S., Damian, A., Hu, S., Ravi, N., & Rudin, C.. (2020). PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models.
[4] Zhang, R., Isola, P., Efros, A., Shechtman, E., & Wang, O.. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
