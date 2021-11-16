# image_processing.py
# Chris McKay
# v1.0 2021-11-14
# script to randomly apply transformations.py to each image in configured folder

import random
from os import listdir
from os.path import isfile, join
from PIL import Image
from inspect import getmembers, isfunction
import transformations as t
from .. import config
from .. import time_prediction


in_path = '/model_pictures/originals_untagged/'
out_path = '../model_pictures/originals_modified/'
transform_quantity = 6

# get list of files in path
file_list = [f for f in listdir(config.FILE_IN_PATH) if isfile(join(config.FILE_IN_PATH, f))]

# get list of modules in transformations.py that start with im_
modules = getmembers(t, isfunction)     # find members of transformations that are functions
modules = [i[0] for i in modules]       # get only the function name - first value in each tuple
im_modules = [x for x in modules if x.startswith('im_')]    # filter only for im_ prefixes
morph_modules = [x for x in modules if x.startswith('morph_')]    # filter only for morph_ prefixes

# open outputfile to write to
f = open('transformations.txt', 'w')

tp = time_prediction.TimePredictor(len(file_list))

# for each file in the filelist
for idx, file in enumerate(file_list):

    # open the original file
    orig_img = Image.open(config.FILE_IN_PATH + file)

    pic_name = file.rsplit('.', 1)[0]

    # perform transform_quantity transformations to each image
    for iterator in range(1, transform_quantity + 1, 1):

        # select a number of morphs, picking at least 1
        number_of_morphs = random.randint(1, len(morph_modules))
        selected_morphs = random.sample(morph_modules, k=number_of_morphs)

        # randomly pick between a min of 1 and "all" transformations and make list to iterate function calls
        number_of_transformations = random.randint(1, min(1, int(len(im_modules) * 0.6)))
        selected_transformations = random.sample(im_modules, k=number_of_transformations)

        selected_modifications = selected_morphs + selected_transformations

        write_line = "\n" + pic_name + '_' + str(iterator) + ':' + str(selected_modifications)
        print(write_line)
        f.write(write_line)

        # reinitialize image
        modified_img = orig_img

        # iterate through randomly selected transformation functions
        for transform in selected_modifications:

            # dynamically call function by each string in list
            method_to_call = getattr(t, transform)
            modified_img = method_to_call(modified_img)

        # save file with original name plus iteration number
        modified_img.save(config.FILE_OUT_PATH +
                    pic_name + '_' +
                    str(len(selected_modifications)) + '_' +
                    str(iterator) + '.jpg')

        # update prediction every 10% done
        if (idx / len(file_list)) % 0.1 == 0:
            tp.predict(idx+1)








