# pytorch-unet-segmentation

**Members** : <a href="https://github.com/PyeongKim">PyeongEun Kim</a>, <a href="https://github.com/juhlee">JuHyung Lee</a>, <a href="https://github.com/mijeongl"> MiJeong Lee </a>

**Supervisors** : <a href="https://github.com/utkuozbulak">Utku Ozbulak</a>, Wesley De Neve

## Description

This project aims to implement biomedical image segmentation with the use of U-Net model. The below image briefly explains the output we want:

<p align="center">
<img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/segmentation_image.jpg">


The dataset we used is Transmission Electron Microscopy (ssTEM) data set of the Drosophila first instar larva ventral nerve cord (VNC), which is dowloaded from [ISBI Challenge: Segmentation of of neural structures in EM stacks](http://brainiac2.mit.edu/isbi_challenge/home)

The dataset contains 30 images (.png) of size 512x512 for each train, train-labels and test.


## Table of Content

* [Dataset](#dataset)

* [Preprocessing](#preprocessing)

* [Model](#model)

* [Loss function](#lossfunction)

* [Post-processing](#postprocessing)

* [Results](#results)

* [Dependency](#dependency)

* [References](#references)



## Dataset <a name="dataset"></a>

```ruby
class SEMDataTrain(Dataset):

    def __init__(self, image_path, mask_path, in_size=572, out_size=388):
        """
        Args:
            image_path (str): the path where the image is located
            mask_path (str): the path where the mask is located
            option (str): decide which dataset to import
        """
        # All file names
	# Lists of image path and list of labels
        # Calculate len
        # Calculate mean and stdev

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
          # Flip 
          # Gaussian_noise
          # Uniform_noise
          # Brightness
          # Elastic distort {0: distort, 1:no distort}
          # Crop the image
          # Pad the image
          # Sanity Check for Cropped image
          # Normalize the image

          # Add additional dimension
          # Convert numpy array to tensor
        
        #Augmentation on mask
          # Flip same way with image
          # Elastic distort same way with image
          # Crop the same part that was cropped on image
          # Sanity Check
          # Normalize the mask to 0 and 1
      
        # Add additional dimension
        # Convert numpy array to tensor

        return (img_as_tensor, msk_as_tensor)

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """

```

## Preprocessing <a name="preprocessing"></a>

We preprocessed the images for data augmentation. Following preprocessing are :
   * Flip
   * Gaussian noise
   * Uniform noise
   * Brightness
   * Elastic deformation
   * Crop
   * Pad 
   
#### Image Augmentation


<p align="center">
  <img width="250" height="250" src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/original.png"> <br />Original Image</td>
</p>


<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="4"><strong>Image</td>
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
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_10"> <br />Standard Deviation: 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_50"> <br />Standard Deviation: 50</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/gn_100"> <br />Standard Deviation: 100</td>
   		</tr>
		<tr>
			<td width="19%" align="center"> Uniform noise </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/uniform_10"> <br />Intensity: 10 </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/un_50"> <br />Intensity: 50</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/un_100"> <br />Intensity: 100</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Brightness </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/bright_10"> <br />Intensity: 10</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/br_50.png"> <br />Intensity: 20</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/br_100.png"> <br />Intensity: 30</td>
		</tr>
      		</tr>
		<tr>
			<td width="19%" align="center"> Elastic deformation </td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_10.png"> <br />Random Deformation: 1</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_34.png"> <br />Random Deformation: 2</td>
			<td width="27%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/ed_50.png"> <br />Random Deformation: 3</td>
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
			<td width="25%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/c_rb"> <br /> Right Bottom</td>
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


## Model <a name="model"></a>

#### Architecture

We have same structure as U-Net Model architecture but we made a small modification to make the model smaller.

![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/UNet_custom_parameter.png)

## Loss function <a name="lossfunction"></a>

We used a loss function where pixel-wise softmax is combined with cross entropy.

#### Softmax
![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/softmax(1).png)

#### Cross entropy
![image](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/cross%20entropy(1).png)

## Post-processing <a name="postprocessing"></a>
In attempt of reducing the loss, we did a post-processing on the prediction results. We applied the concept of watershed segmentation in order to point out the certain foreground regions and remove regions in the prediction image which seem to be noises.

![postprocessing](https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/postprocess.png)

The numbered images in the figure above indicates the stpes we took in the post-processing. To name those steps in slightly more detail:

	* 1. Convertion into grayscale
	* 2. Conversion into binary image
	* 3. Morphological transformation: Closing
	* 4. Determination of the certain background
	* 5. Calculation of the distance
	* 6. Determination of the certain foreground
	* 7. Determination of the unknown region
	* 8. Application of watershed
	* 9. Determination of the final result

### Conversion into grayscale 

The first step is there just in case the input image has more than 1 color channel (e.g. RGB image has 3 channels) 

### Conversion into binary image

Convert the gray-scale image into binary image by processing the image with a threshold value: pixels equal to or lower than 127 will be pushed down to 0 and greater will be pushed up to 255. Such process is compulsory as later transformation processes takes in binary images.

### Morphological transformation: Closing.

We used **morphologyEX()** function in cv2 module which removes black noises (background) within white regions (foreground).
	
### Determination of the certain background

We used **dilate()** function in cv2 module which emphasizes/increases the white region (foreground). By doing so, we connect detached white regions together - for example, connecting detached cell membranes together - to make ensure the background region.

### Caculation of the distance

This step labels the foreground with a color code: ![#ff0000](https://placehold.it/15/ff0000/000000?text=+) red color indicates farthest from the background while ![#003bff](https://placehold.it/15/003bff/000000?text=+) blue color indicates closest to the background.

### Determination of the foreground

Now that we have an idea of how far the foreground is from the background, we apply a threshold value to decide which part could surely be the foreground.

The threshold value is the maximum distance (calculated from the previous step) multiplied by a hyper-parameter that we have to manually tune. The greater the hyper-parameter value, the greater the threshold value, and therefore we will get less area of certain foreground. 

### Determination of the unknown region

From previous steps, we determined sure foreground and background regions. The rest will be classified as *'unknown'* regions.

### Label the foreground: markers

We applied **connectedComponents()** function from the cv2 module on the foreground to label the foreground regions with color to distinguish different foreground objects. We named it as a 'marker'.

### Application of watershed and Determination of the final result

After applying **watershed()** function from cv2 module on the marker, we obtained an array of -1, 1, and many others. 

	* -1 = Border region that distinguishes foreground and background
	*  1 = Background region

To see the result, we created a clean white page of the same size with the input image. then we copied all the values from the watershed result to the white page except 1, which means that we excluded the background.

## Results <a name="results"></a>

<table style="width:99%">
	<tr> 
		<th>Optimizer</th>
	    	<th>Learning Rate</th>
	    	<th>Lowest Loss</th>
	    	<th>Epoch</th>
		<th>Highest Accuracy</th>
	    	<th>Epoch</th>
	</tr>
	<tr>
		<th rowspan="3">SGD</th>
		<td align="center">0.001</td>
		<td align="center">0.196972</td>
		<td align="center">1445</td>
		<td align="center">0.921032</td>
		<td align="center">1855</td>
	</tr>
	<tr>
		<td align="center">0.005</td>
		<td align="center">0.205802</td>
		<td align="center">1815</td>
		<td align="center">0.918425</td>
		<td align="center">1795</td>
	</tr>
	<tr>
		<td align="center">0.01</td>
		<td align="center">0.193328</td>
		<td align="center">450</td>
		<td align="center">0.922908</td>
		<td align="center">450</td>
	</tr>
	<tr>
		<th rowspan="3">RMS_prop</th>
		<td align="center">0.0001</td>
		<td align="center">0.203431</td>
		<td align="center">185</td>
		<td align="center">0.924543</td>
		<td align="center">230</td>
	</tr>
	<tr>
		<td align="center">0.0002</td>
		<td align="center">0.193456</td>
		<td align="center">270</td>
		<td align="center">0.926245</td>
		<td align="center">500</td>
	</tr>
	<tr>
		<td align="center">0.001</td>
		<td align="center">0.268246</td>
		<td align="center">1655</td>
		<td align="center">0.882229</td>
		<td align="center">1915</td>
	</tr>
	<tr>
		<th rowspan="3">Adam</th>
		<td align="center">0.0001</td>
		<td align="center">0.194180</td>
		<td align="center">140</td>
		<td align="center">0.924470</td>
		<td align="center">300</td>
	</tr>
	<tr>
		<td align="center">0.0005</td>
		<td align="center">0.185212</td>
		<td align="center">135</td>
		<td align="center">0.925519</td>
		<td align="center">135</td>
	</tr>
	<tr>
		<td align="center">0.001</td>
		<td align="center">0.222277</td>
		<td align="center">165</td>
		<td align="center">0.912364</td>
		<td align="center">180</td>
	</tr>
		
</table>       


We chose the best learning rate that fits the optimizer based on **how fast the model converges to the lowest error**. In other word, the learning rate should make model to reach optimal solution in shortest epoch repeated. However, the intersting fact was that the epochs of lowest loss and highest accuracy were not corresponding. This might be due to the nature of loss function (Loss function is log scale, thus an extreme deviation might occur). For example, if the softmax probability of one pixel is 0.001, then the -log(0.001) would be 1000 which is a huge value that contributes to loss.
For consistency, we chose to focus on accuracy as our criterion of correctness of model. 



<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="3"><strong>Accuracy and Loss Graph</td>
	    </tr>
		<tr>
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/SGD_graph.png"> </td> 
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/RMS_graph.png"> </td>
			<td width="33%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/Adam_graph.png"> </td>
		</tr>
		<tr>
			<td align="center">SGD<br />(lr=0.01,momentum=0.99)</td>
			<td align="center">RMS prop<br />(lr=0.0002)</td>
			<td align="center">Adam<br />(lr=0.0005)</td>
      		</tr>
	</tbody>
</table>       
We used two different optimizers (SGD, RMS PROP, and Adam). In case of SGD the momentum is manually set (0.99) whereas in case of other optimizers (RMS Prop and Adam) it is calculated automatically. 

### Model Downloads

Model trained with SGD can be downloaded via **dropbox**:
https://www.dropbox.com/s/ge9654nhgv1namr/model_epoch_2290.pwf?dl=0


Model trained with RMS prop can be downloaded via **dropbox**:
https://www.dropbox.com/s/cdwltzhbs3tiiwb/model_epoch_440.pwf?dl=0


Model trained with Adam can be downloaded via **dropbox**:
https://www.dropbox.com/s/tpch6u41jrdgswk/model_epoch_100.pwf?dl=0




### Example

<p align="center">
  <img width="250" height="250" src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_img.png"> <br /> Input Image</td>
</p>

<table border=0 width="99%" >
	<tbody> 
    <tr>		<td width="99%" align="center" colspan="5"><strong>Results comparsion</td>
	    </tr>
		<tr>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_mask.png"> </td>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_RMS.png"> </td>
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_SGD.png"></td> 
			<td width="24%" align="center"> <img src="https://github.com/ugent-korea/pytorch-unet-segmentation/blob/master/readme_images/validation_Adam.png"> </td>
		</tr>
		<tr>
			<td align="center">original image mask</td>
			<td align="center">RMS prop optimizer <br />(Accuracy 92.48 %)</td>
			<td align="center">SGD optimizer <br />(Accuracy 91.52 %)</td>
			<td align="center">Adam optimizer <br />(Accuracy 92.55 %)</td>
      		</tr>
	</tbody>
</table>       

## Dependency <a name="dependency"></a>

Following modules are used in the project:

    * python >= 3.6
    * numpy >= 1.14.5
    * torch >= 0.4.0
    * PIL >= 5.2.0
    * scipy >= 1.1.0
    * matplotlib >= 2.2.2
   

## References <a name="references"></a> :

[1] O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation, http://arxiv.org/pdf/1505.04597.pdf

[2] P.Y. Simard, D. Steinkraus, J.C. Platt. Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis, http://cognitivemedium.com/assets/rmnist/Simard.pdf
