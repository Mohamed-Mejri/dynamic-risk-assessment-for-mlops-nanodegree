"""Module for helpful functions

Author: Mohamed Mejri
Date: Feb 2023
"""
import os
import shutil


def create_dir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def remove_all_files(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path) 
    elif os.path.isdir(path):
        shutil.rmtree(path)
    else:
        raise ValueError("file {} is not a file or dir.".format(path))
