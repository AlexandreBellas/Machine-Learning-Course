# This module is responsible for loading the data set and apply ao the data augmentation operations

import os
import os.path
from skimage import data, io, filters

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
