# pytorch-unet-segmentation

**Members** : PyeongEun Kim, JuHyung Lee, MiJeong Lee

**Supervisor** : Utku Ozbulak, Wesley De Neve

## Description


This project aims to implement biomedical image segmentation with the use of U-Net model. The below image briefly explains the output we want:

<p align="center">
<img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/segmentation_image.jpg">


The dataset we used is Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC), which is dowloaded from [ISBI Challenge: Segmentation of of neural structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/home)

The dataset contains 30 images (.png) of size 512x512 for each train, train-labels and test.

The folder structure of this project is:

```
pytorch-unet-segmentation
   - data
       - train
           - images
           - masks
       - test
           - images
   - src
       - result_images
           - train
	       - epoch_1
	           - train_1.png
		   - ...
	       - epoch_2
	       - ...
	    - test
	        - (same structure with that of train)	    
       - dataset.py
       - main.py
       - augmentation.py
       - mean_std.py
       - modules.py
```

Purposes of the python files listed in the folder structure will be explained throughout this readme.

## Table of Content

* [Pipeline](#pipeline)

* [Preprocessing](#preprocessing)

* [Model](#model)

* [Loss function](#lossfunction)

* [Results](#results)

* [Dependency](#dependency)


## Pipeline <a name="pipeline"></a>


### Dataset

```ruby
from torch.utils.data.dataset import Dataset
class SEMDataTrain(Dataset):
    def __init__(self):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data

        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        return (img_as_tensor, msk_as_tensor)

```
This is a dataset class we used. In the dataset, it contains three functions.
  * \__intit\__ : Intialization 
  * \__getitem\__ : Reads the images and preprocess on the images are accompolished. 
  * \__len\__ : Counts the number of images. 


Before reading the images, in \__init\__ function with the parameter, image_path, list of image names and image labels are collected with the module called **glob**. Then in \__getitem\__ function, with the module **PIL**, the images in the list of image names are read and converted into numpy array. 

```ruby
import numpy as np
from PIL import Image
import glob
from torch.utils.data.dataset import Dataset
class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # all file names
	# lists of image path and list of labels
        # Calculate len
        self.data_len = len(self.mask_arr)
        # calculate mean and stdev

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data

        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
        """
        # GET IMAGE
        """
        #Augmentation on image
          # flip 
          # Gaussian_noise
          # uniform_noise
          # Brightness
          # Elastic distort {0: distort, 1:no distort}
          # Crop the image
          # Pad the image
          # Sanity Check for Cropped image
          # Normalize the image
        """
          # add additional dimension
          # Convert numpy array to tensor
        
        """
        #Augmentation on mask
          # flip same way with image
          # Elastic distort same way with image
          # Crop the same part that was cropped on image
          # Sanity Check
          # Normalize the mask to 0 and 1
      
        # add additional dimension
        # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len
```

### Preprocessing <a name="preprocessing"></a>

Preprocessing is done on the images for data augmentation. Following preprocessing are accomplished:
   * Flip
   * Gaussian noise
   * Uniform noise
   * Brightness
   * Elastic deformation
   * Crop
   * Pad 
   
#### Image Augmentation

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="4"><strong>Image</td>
		</tr>
		<tr>
			<td width="19%" align="center"> Original Image </td>
			<td width="27%" align="center"> </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/original.png"> </td>
			<td width="27%" algin="center"> </td>
		</tr>
		</tr>
			<td width="19%" align="center"> Flip  </td> 
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_vert"> <br />Vertical  </td> 
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_hori">  <br />Horizontal</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_both"> <br />Both</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Gaussian noise </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_10"> <br />standard deviation 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_50"> <br />standard deviation 50</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_100"> <br />standard deviation 100</td>
   		</tr>
		<tr>
			<td width="19%" align="center"> Uniform noise </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/uniform_10"> <br />Intensity 10 </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/un_50"> <br />Intensity 50</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/un_100"> <br />Intensity 100</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Brightness </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/bright_10"> <br />Intensity 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/br_50.png"> <br />Intensity 20</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/br_100.png"> <br />Intensity 30</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Elastic deformation </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_10.png"> <br />random deformation 1</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_34.png"> <br />random deformation 2</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_50.png"> <br />random deformation 3</td>
		</tr>
		</tr>
	</tbody>
</table>       

#### Crop and Pad

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="4"><strong>Crop</td>
	    </tr>
		<tr>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/c_lb"> <br />  Left Bottom </td>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/c_lt"> <br /> Left Top</td> 
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/c_rb"> <br /> Right bottom</td>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/c_rt"> <br /> Right Top</td> 
		</tr>
      		</tr>
	</tbody>
</table>         

Padding process is compulsory after the cropping process as the image has to fit the input size of the U-Net model. 

In terms of the padding method, **symmetric padding** was done in which the pad is the reflection of the vector mirrored along the edge of the array. We selected the symmetric padding over several other padding options because it reduces the loss the most. 

To help with observation, a ![#ffff00](https://placehold.it/15/ffff00/000000?text=+) 'yellow border' is added around the original image: outside the border indicates symmetric padding whereas inside indicates the original image.

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="4"><strong>Pad</td>
	    </tr>
		<tr>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/p_lb.PNG"> <br />  Left Bottom </td>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/p_lt.PNG"> <br /> Left Top</td> 
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/p_rb.PNG"> <br /> Right bottom</td>
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/p_rt.PNG"> <br /> Right Top</td> 
		</tr>
      		</tr>
	</tbody>
</table>         


### Model <a name="model"></a>
#### Architecture

We have same structure as U-Net Model architecture but we made a small modification to make the model smaller.

![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/UNet_custom_parameter.png)

### Loss function <a name="lossfunction"></a>
We used a loss function where pixel-wise softmax is combined with cross entropy.

#### Softmax
![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/cross%20entropy(1).png)

#### Cross entropy
![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/softmax(1).png)

### Results <a name="results"></a>
To be add

### Dependency <a name="dependency"></a>
    * python >= 3.6
    * numpy >= 1.14.5
    * torch >= 0.4.0
    * PIL >= 5.2.0
    * scipy >= 1.1.0
    

# References :

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf

[2] P.Y. Simard, D. Steinkraus, J.C. Platt. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, http://cognitivemedium.com/assets/rmnist/Simard.pdf
