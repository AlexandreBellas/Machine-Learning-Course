from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure
import numpy as np


def get_dataset_hogs(dataset, dataset_name):
    """
    get_dataset_hogs gets the hog feature for each image in the dataset
    :param dataset: the data set to be used. It must have the structure
        [
            [person_1_face1, person_1_face2, person_1_face3],
            [person_2_face1, person_2_face2, person_2_face3],
            [person_3_face1, person_3_face2, person_3_face3]

        ]
    :return: a slice of slices containing the feature hogs following the schema below
        [
            [person_1_face1_hog, person_1_face2_hog, person_1_face3_hog],
            [person_2_face1_hog, person_2_face2_hog, person_2_face3_hog],
            [person_3_face1_hog, person_3_face2_hog, person_3_face3_hog]

        ]
    """
    return [[get_hog(img, dataset_name) for img in person] for person in dataset]


def save_hogs_dataset(dataset, file):
    """
    save_hogs_dataset saves the hog vectors extracted from a database in the given file
    :param dataset: a slice of slices containing the feature hogs following the schema below
        [
            [person_1_face1_hog, person_1_face2_hog, person_1_face3_hog],
            [person_2_face1_hog, person_2_face2_hog, person_2_face3_hog],
            [person_3_face1_hog, person_3_face2_hog, person_3_face3_hog]

        ]
    :param file: file to save the hog features
    """
    np.save(file, dataset)

def read_hogs_dataset(file):
    """
    read_hog_dataset reads the hog features of a dataset from the given file
    :param file: file to read the features from
    :return: a slice of slices containing the feature hogs following the schema below
        [
            [person_1_face1_hog, person_1_face2_hog, person_1_face3_hog],
            [person_2_face1_hog, person_2_face2_hog, person_2_face3_hog],
            [person_3_face1_hog, person_3_face2_hog, person_3_face3_hog]

        ]
    """
    return np.load(file)

def get_hog(img, dataset_name, print_hog=False):
    """
    get_hog returns the hog feature for the given image
    @param img: the image to extract the hog feature
    @return: a slice containing the image hog feature
    """

    # Parameters
    if dataset_name == 'icmc_hogs':
        pixels_per_cell = (32,32)
    elif dataset_name == 'orl_hogs':
        pixels_per_cell = (8,8)
    orientations = 8
    cell_per_block = (2,2)
    block_norm = 'L2'

    if print_hog:
        fd, hog_image = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cell_per_block, visualize=True, multichannel=False, block_norm=block_norm)

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

    else:
        fd = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cell_per_block, visualize=False, multichannel=False, block_norm=block_norm)

    return fd