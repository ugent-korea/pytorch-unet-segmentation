# Bring in different python files
from check_accuracy import *
from post_processing import *
# Importing actual modules
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import glob as gl


def accuracy_compare(prediction_folder, true_mask_folder):

    ''' Output average accuracy of all prediction results and their corresponding true masks.
    Args
        prediction_folder : folder of the prediction results
        true_mask_folder : folder of the corresponding true masks
    Returns
        a tuple of (original_accuracy, posprocess_accuracy)
    '''

    # Bring in the images
    all_prediction = gl.glob(prediction_folder)
    all_mask = gl.glob(true_mask_folder)

    # Initiation
    num_files = len(all_prediction)
    count = 0
    postprocess_acc = 0
    original_acc = 0

    while count != num_files:

        # Prepare the arrays to be further processed.
        prediction_processed = postprocess(all_prediction[count])
        prediction_image = Image.open(all_prediction[count])
        mask = Image.open(all_mask[count])

        # converting the PIL variables into numpy array
        prediction_np = np.asarray(prediction_image)
        mask_np = np.asarray(mask)

        # Calculate the accuracy of original and postprocessed image
        postprocess_acc += accuracy_check(mask_np, prediction_processed)
        original_acc += accuracy_check(mask_np, prediction_np)
        # check individual accuracy
        print(str(count) + 'th post acc:', accuracy_check(mask_np, prediction_processed))
        print(str(count) + 'th original acc:', accuracy_check(mask_np, prediction_np))

        # Move onto the next prediction/mask image
        count += 1

    # Average of all the accuracies
    postprocess_acc = postprocess_acc / num_files
    original_acc = original_acc / num_files

    return (original_acc, postprocess_acc)

# Experimenting
if __name__ == '__main__':
    '''
    predictions = 'result/*.png'
    masks = '../data/val/masks/*.png'

    result = accuracy_compare(predictions, masks)
    print('Original Result :', result[0])
    print('Postprocess result :', result[1])
    '''
