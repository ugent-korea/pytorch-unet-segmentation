import numpy as np
from matplotlib import pyplot as plt


def postprocess(image_path):
    ''' postprocessing of the prediction output
    Args
        image_path : path of the image
    Returns
        watershed_grayscale : numpy array of postprocessed image (in grayscale)
    '''

    # Bring in the image
    img_original = cv2.imread(image_path)
    img = cv2.imread(image_path)

    # In case the input image has 3 channels (RGB), convert to 1 channel (grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Use threshold => Image will have values either 0 or 255 (black or white)
    ret, bin_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove Hole or noise through the use of opening, closing in Morphology module
    kernel = np.ones((1, 1), np.uint8)
    kernel1 = np.ones((3, 3), np.uint8)

    # remove noise in
    closing = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel, iterations=1)

    # make clear distinction of the background
    # Incerease/emphasize the white region.
    sure_bg = cv2.dilate(closing, kernel1, iterations=1)

    # calculate the distance to the closest zero pixel for each pixel of the source.
    # Adjust the threshold value with respect to the maximum distance. Lower threshold, more information.
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown is the region of background with foreground excluded.
    unknown = cv2.subtract(sure_bg, sure_fg)

    # labelling on the foreground.
    ret, markers = cv2.connectedComponents(sure_fg)
    markers_plus1 = markers + 1
    markers_plus1[unknown == 255] = 0

    # Appy watershed and label the borders
    markers_watershed = cv2.watershed(img, markers_plus1)

    # See the watershed result in a clear white page.
    img_x, img_y = img_original.shape[0], img_original.shape[1]  # 512x512
    white, white_color = np.zeros((img_x, img_y, 3)), np.zeros((img_x, img_y, 3))
    white += 255
    white_color += 255
    # 1 in markers_watershed indicate the background value
    # label everything not indicated as background value
    white[markers_watershed != 1] = [0, 0, 0]  # grayscale version
    white_color[markers_watershed != 1] = [255, 0, 0]  # RGB version

    # Convert to numpy array for later processing
    white_np = np.asarray(white)  # 512x512x3
    watershed_grayscale = white_np.transpose(2, 0, 1)[0, :, :]  # convert to 1 channel (grayscale)
    img[markers_watershed != 1] = [255, 0, 0]

    return watershed_grayscale

    '''
    Visualizing all the intermediate processes

    images = [img_original, gray,bin_image, closing, sure_bg,  dist_transform, sure_fg, unknown, markers, markers_watershed, white_color, white, img]
    titles = ['Original', '1. Grayscale','2. Binary','3. Closing','Sure BG','Distance','Sure FG','Unknown','Markers', 'Markers_Watershed','Result', 'Result gray','Result Overlapped']
    CMAP = [None, 'gray', 'gray','gray','gray',None,'gray','gray',None, None, None, None,'gray']


    for i in range(len(images)):
        plt.subplot(4,4,i+1),plt.imshow(images[i], cmap=CMAP[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

    plt.show()
    '''


if __name__ == '__main__':
    from PIL import Image

    print(postprocess('../data/train/masks/25.png'))
