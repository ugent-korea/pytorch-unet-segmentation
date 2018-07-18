# pytorch-unet-segmentation

## Description
The project involves implementing biomedical image segmentation based on U-Net. 

##### Members : PyeongEun Kim, JuHyung Lee, MiJeong Lee

##### Supervisors : Utku Ozbulak

* [Prerequisite](#prerequisite)

* [Pipeline](#pipeline)

* [Preprocessing](#preprocessing)

* [Model](#model)

* [Loss function](#lossfunction)

* [Results](#results)

* [Dependency](#dependency)

## Prerequisite <a name="prerequisite"></a>

    * python 3.6
    * numpy 1.14.5
    * torch 0.4.0
    * PIL
    * glob
    
    
## Pipeline <a name="pipeline"></a>

### Dataset

```ruby
from torch.utils.data.dataset import Dataset
class SEMDataTrain(Dataset):
    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
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

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len
        
if __name__ == "__main__":

    custom_mnist_from_file_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    custom_mnist_from_file_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')

    imag_1 = custom_mnist_from_file_train.__getitem__(0)
    imag_2 = custom_mnist_from_file_test.__getitem__(0)
    print(imag_2.size())
```
This is a dataset class we used. In the dataset, it contains three functions.
  * \__intit\__ : Intialization happens and determines which datasets to import and read. 
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
        self.mask_arr = glob.glob(str(mask_path) + "/*")
        self.image_arr = glob.glob(str(image_path) + str("/*"))
        self.in_size, self.out_size = in_size, out_size
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
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        # img_as_img.show()
        img_as_np = np.asarray(img_as_img)
        """
        Augmentation
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
        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len
if __name__ == "__main__":

    custom_mnist_from_file_train = SEMDataTrain(
        '../data/train/images', '../data/train/masks')
    custom_mnist_from_file_test = SEMDataTest(
        '../data/test/images/', '../data/test/masks')

    imag_1 = custom_mnist_from_file_train.__getitem__(0)
    imag_2 = custom_mnist_from_file_test.__getitem__(0)
    print(imag_2.size())
```

### Preprocessing <a name="preprocessing"></a>

### Model <a name="model"></a>
To be added

### Loss function <a name="lossfunction"></a>
To be added

### Results <a name="results"></a>
To be added

### Dependency <a name="dependency"></a>

# References :

O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf
