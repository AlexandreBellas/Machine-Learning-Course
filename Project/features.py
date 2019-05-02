from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure



def get_hog(img, print_hog=False):
    """
    get_hog returns the hog feature for the given image
    @param img: the image to extract the hog feature
    @return: a slice containing the image hog feature
    """

    fd, hog_image = hog(img, orientations=8, pixels_per_cell=(6, 6),
                        cells_per_block=(2, 2), visualize=True, multichannel=False, block_norm='L2')

    if print_hog:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(img, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range='image')

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return fd
