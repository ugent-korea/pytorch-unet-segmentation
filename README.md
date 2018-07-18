# pytorch-unet-segmentation

## Description
The project involves implementing biomedical image segmentation based on U-Net. 

##### Members : PyeongEun Kim, JuHyung Lee, MiJeong Lee

##### Supervisors : Utku Ozbulak

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


### Reading images
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
        Augmentation on image
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
        Augmentation on mask
          # flip same way with image
          # Elastic distort same way with image
          # Crop the same part that was cropped on image
          # Sanity Check
          # Normalize the mask to 0 and 1
        """
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

Preprocessing is done on the images for data augmentation. Following preprocessing are accomplished.
   * Flip
   * Gaussian noise
   * Uniform noise
   * Brightness
   * Elastic deformation


<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="4"><strong>Image</td>
		</tr>
		<tr>
			<td width="19%" align="center"> Original Image </td>
			<td width="27%" align="center" colspan= "1" >
			<td width="27%" align="center" colspan= "1" > <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/original_image"> </td> 
			<td width="27%" align="center" colspan= "1" ></td> 
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Flip  </td> 
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_vert"> <br />Vertical  </td> 
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_hori">  <br />Horizontal</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/flip_both"> <br />Both</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Gaussian noise </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gaus_10"> <br />standard deviation 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gaus_20"> <br />standard deviation 20</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gaus_30"> <br />standard deviation 30</td>
   		</tr>
		<tr>
			<td width="19%" align="center"> Uniform noise </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/uniform_10"> <br />Intensity 10 </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/uniform_20"> <br />Intensity 20</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/uniform_30"> <br />Intensity 30</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Brightness </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/bright_10"> <br />Intensity 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/bright_20"> <br />Intensity 20</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/bright_30"> <br />Intensity 30</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Elastic deformation </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/elastic_1"> <br />random deformation 1</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/elastic_2"> <br />random deformation 2</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/elastic_3"> <br />random deformation 3</td>
		</tr>
		</tr>
	</tbody>
</table>       



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



### Model <a name="model"></a>
To be added


### Loss function <a name="lossfunction"></a>
To be added


### Results <a name="results"></a>
To be added


### Dependency <a name="dependency"></a>
    * python >= 3.6
    * numpy >= 1.14.5
    * torch >= 0.4.0
    * PIL >=
    * glob >= 
    * scipy >=
    

# References :

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf

[2] P.Y. Simard, D. Steinkraus, J.C. Platt. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, http://cognitivemedium.com/assets/rmnist/Simard.pdf
