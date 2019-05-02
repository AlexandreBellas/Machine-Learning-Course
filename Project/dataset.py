# This module is responsible for loading the data set and apply ao the data augmentation operations

import os
import os.path

from copy import deepcopy

import numpy as np
from scipy import ndimage

from skimage import data, io, filters, exposure
from skimage.util import random_noise, img_as_uint, invert
from skimage.transform import rotate

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def print_data():
    print("Hello from dataset")


def get_faces(path='.'):
    """
    get_faces returns the faces in the data set
    @param path: the data set directory to read the data from. Each person must have a separate folder with
    its corresponding faces
    @return: a slice with each index pointing to a slice containing a person's faces
    """

    dataset = []

    # Get all files in directory except hidden files
    people = [file for file in os.listdir(path) if not file.startswith('.')]

    for person in people:
        person_path = os.path.join(path, person)

        faces = os.listdir(person_path)
        faces.sort()
        dataset.append([])

        for face in faces:
            file = os.path.join(person_path, face)
            img = io.imread(file)
            dataset[-1].append(img)

    return dataset


def enhance_data(dataset):
    """
    enhance_data creates synthetic data for a data set that doesn't have 10 faces per person
    It applies data augmentation techniques to a person's first face to generate new ones and append them into
    the person slice until it reaches 10 images
    @param dataset: the original data set
    @return: the data set after augmentation
    """

    augmented_dataset = deepcopy(dataset)
    for person in augmented_dataset:
        while len(person) < 10:
            # TODO Use better techniques of data augmentation
            # https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage

            # Noise insertion => needed to convert to uint16 type for saving in a file
            img_noised = random_noise(person[0], mode='pepper', amount=0.2)
            img_noised = exposure.rescale_intensity(img_noised, in_range='image', out_range='uint16')
            img_noised = img_noised.astype(np.uint16)
            person.append(img_noised)

            # Inversed colors
            img_inversed = np.invert(person[0])
            person.append(img_inversed)

            # Rotation by 20 degrees backward => needed to convert to uint16 type for saving in a file
            img_rotated_backward = rotate(person[0], -20, mode='edge')
            img_rotated_backward = exposure.rescale_intensity(img_rotated_backward, in_range='image', out_range='uint16')
            img_rotated_backward = img_rotated_backward.astype(np.uint16)
            person.append(img_rotated_backward)

            # Constrast changed
            v_min, v_max = np.percentile(person[0], (0.1, 38.0))
            img_better_constrast = exposure.rescale_intensity(person[0], in_range=(v_min, v_max))
            person.append(img_better_constrast)

            # Logarithmic correction
            img_log_correction = exposure.adjust_log(person[0], gain=1.1)
            person.append(img_log_correction)

            # Sigmoid correction
            img_sigmoid_correction = exposure.adjust_sigmoid(person[0])
            person.append(img_sigmoid_correction)

            # Horizontal flip
            img_horizontal_flip = person[0][:, ::-1]    
            person.append(img_horizontal_flip)

            # Vertical flip
            img_vertical_flip = person[0][::-1, :]    
            person.append(img_vertical_flip)    

            # Blur image => not working! There is a problem in the parameters of the function below...
            img_blured = ndimage.uniform_filter(person[0], size=10)
            person.append(img_blured)

    return augmented_dataset


def save_faces(dataset, path):
    """
    save_faces persists the data set. A new directory is created for each person in which the person's faces are
    stored as png files
    @param dataset: the data set to be saved. It must have the structure
        [
            [person_1_face1, person_1_face2, person_1_face3],
            [person_2_face1, person_2_face2, person_2_face3],
            [person_3_face1, person_3_face2, person_3_face3]

        ]
    @param path: the already existed directory where the data set will be saved
    @return:
    """

    for person, person_id in zip(dataset, range(1, len(dataset)+1)):
        person_path = os.path.join(path, "p" + str(person_id))

        try:
            os.makedirs(person_path, exist_ok=True)
        except OSError:
            print("ERROR creating the directory %s" % person_path)
            raise OSError

        for face, file_id in zip(person, range(1, len(person)+1)):
            image_path = os.path.join(person_path, str(file_id) + '.png')
            io.imsave(image_path, face)
