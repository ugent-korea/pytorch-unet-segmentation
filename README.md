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
```

### Preprocessing <a name="preprocessing"></a>

Preprocessing is done on the images for data augmentation. Following preprocessing are accomplished.
   * Flip
   * Gaussian noise
   * Uniform noise
   * Brightness
   * Elastic deformation

Preprocess   | Image 1 | Image 2 | Image 3 |
------------ | ------------- | ------------ | ------------- |
Original Images | Content from cell 2|  | |
Flip | Content in the second column |  | |
Gaussian noise | Content from cell 2 |  | |
Uniform noise | Content in the second column  |  | |
Brightness | Content from cell 2 |  | |
Elastic deformation | Content in the second column|  | |

   
```ruby
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from random import randint


def add_elastic_transform(image, alpha, sigma, seed=None):
    """
    Args:
        image : numpy array of image
        alpha : α is a scaling factor
        sigma :  σ is an elasticity coefficient
        random_state = random integer
        Return :
        image : elastically transformed numpy array of image
    """

    if seed is None:
        seed = randint(1, 100)
        random_state = np.random.RandomState(seed)
    else:
        random_state = np.random.RandomState(seed)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape), seed


def flip(image, option_value):
    """
    Args:
        image : numpy array of image
        option_value = random integer between 0 to 3
    Return :
        image : numpy array of flipped image
    """
    if option_value == 0:
        # vertical
        image = np.flip(image, option_value)
    elif option_value == 1:
        # horizontal
        image = np.flip(image, option_value)
    elif option_value == 2:
        # horizontally and vertically flip
        image = np.flip(image, 0)
        image = np.flip(image, 1)
    else:
        image = image
        # no effect
    return image


def add_gaussian_noise(image, mean=0, std=1):
    """
    Args:
        image : numpy array of image
        mean : pixel mean of image
        standard deviation : pixel standard deviation of image
    Return :
        image : numpy array of image with gaussian noise added
    """
    gaus_noise = np.random.normal(mean, std, image.shape)
    image = image.astype("int16")
    noise_img = image + gaus_noise
    image = ceil_floor_image(image)
    return noise_img


def add_uniform_noise(image, low=-10, high=10):
    """
    Args:
        image : numpy array of image
        low : lower boundary of output interval
        high : upper boundary of output interval
    Return :
        image : numpy array of image with uniform noise added
    """
    uni_noise = np.random.uniform(low, high, image.shape)
    image = image.astype("int16")
    noise_img = image + uni_noise
    image = ceil_floor_image(image)
    return noise_img


def change_brightness(image, value):
    """
    Args:
        image : numpy array of image
        value : brightness
    Return :
        image : numpy array of image with brightness added
    """
    image = image.astype("int16")
    image = image + value
    image = ceil_floor_image(image)
    return image


def ceil_floor_image(image):
    """
    Args:
        image : numpy array of image in datatype int16
    Return :
        image : numpy array of image in datatype uint8 with ceilling(maximum 255) and flooring(minimum 0)
    """
    image[image > 255] = 255
    image[image < 0] = 0
    image = image.astype("uint8")
    return image

```

### Model <a name="model"></a>
To be added


### Loss function <a name="lossfunction"></a>
To be added


### Results <a name="results"></a>
To be added


### Dependency <a name="dependency"></a>


# References :

O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf
P.Y. Simard, D. Steinkraus, J.C. Platt. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, http://cognitivemedium.com/assets/rmnist/Simard.pdf
