# This module is responsible for loading the data set and apply ao the data augmentation operations

import os
import os.path

from copy import deepcopy

from skimage import data, io, filters
from skimage.util import random_noise, img_as_uint

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
            img_noised = random_noise(person[0])
            person.append(img_as_uint(img_noised)) # use img_as_uint since random_noise return a float image

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
