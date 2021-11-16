# tranformations.py
# Carlos Vargas
# v1.0 2021-11-14
# library of transformations to apply to images
# min-max values tunes through experimentation for learning algorithm

import numpy as np
import cv2
import math
import random
from PIL import Image, ImageEnhance, ImageFilter


def random_uniform(min, max):
    random_number = random.random()
    value = max - min
    random_value = value * random_number
    result = min + random_value
    
    return result


def im_brighten(img):
    max = 2
    min = .5
    factor = random_uniform(min, max)
    transformation = ImageEnhance.Brightness(img)
    output = transformation.enhance(factor)
    return output


def im_sharpen(img: Image):
    max = 2
    min = .9
    factor = random_uniform(min, max)
    transformation = ImageEnhance.Sharpness(img)
    output = transformation.enhance(factor)
    return output


def im_blur(img: Image):
    max = 5
    min = 1.4
    factor = random_uniform(min, max)
    blurred_img = img.filter(ImageFilter.GaussianBlur(factor))
    return blurred_img


def morph_mirror(img: Image):
    factor = random.choice([1, 0, -1])
    flipped_image = cv2.flip(np.array(img), factor)
    output = Image.fromarray(flipped_image)
    return output


def im_contrast(img: Image):
    max = 1.5
    min = 0.7
    factor = random_uniform(min, max)
    transformation = ImageEnhance.Contrast(img)
    output = transformation.enhance(factor)
    return output


def morph_rotate(img: Image):
    degrees = random.randint(10, 350)

    # get original and rotated aspect ratios
    aspect_ratio = img.width / img.height

    rotated_image = img.rotate(degrees, expand=1)
    rotated_aspect_ratio = rotated_image.size[0] / rotated_image.size[1]

    # calculate the maximum rectangle possible to inscribe in a rotated rectangle
    an = math.fabs(degrees) * math.pi / 180

    if aspect_ratio < 1:
        total_height = img.size[0] / rotated_aspect_ratio
    else:
        total_height = img.size[1]

    h = total_height / (aspect_ratio * math.fabs(math.sin(an)) + math.fabs(math.cos(an)))
    w = h * aspect_ratio

    # get crop coordinates & crop
    x1 = (rotated_image.width - w) / 2
    y1 = (rotated_image.height - h) / 2
    x2 = rotated_image.width - x1
    y2 = rotated_image.height - y1

    output = rotated_image.crop((x1, y1, x2, y2))
    return output


def im_colourshift(img: Image):
    img = img.convert("RGB")
    d = img.getdata()
    new_image = []
    for item in d:
        # # change all white (also shades of whites)
    # pixels to yellow
        if item[0] in list(range(180, 300)):
            new_image.append((255, 250, 230))
        else:
         new_image.append(item)
        # #update image data
    img.putdata(new_image)
    return img




