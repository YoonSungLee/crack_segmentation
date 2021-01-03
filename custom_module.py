import os
import numpy as np
import cv2 as cv
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from skimage.util import invert
import PIL.Image as pilimg
from plantcv import plantcv as pcv
import seaborn as sns
import random


# image, mask load
image_path = '/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/images'
mask_path = '/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/masks'

image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
image_list = [image for image in image_list if image.endswith('.jpg')]
mask_list = [mask for mask in mask_list if mask.endswith('.jpg')]
image_list = [image for image in image_list if not image.startswith('noncrack')]
mask_list = [mask for mask in mask_list if not mask.startswith('noncrack')]
random.shuffle(image_list)

# max width calculatiton
def max_width(img_path):
    # image read
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    pix = im/255
    pix = np.where(pix>0.5,255,0)
    pix = pix.astype(np.uint8)

    # skeleton
    skeleton = pcv.morphology.skeletonize(mask=pix)

    # prune
    pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=100)

    # distance transform
    dist_transform = cv.distanceTransform(pix, cv.DIST_L2, 5)
    
    # points selection
    result = dist_transform * ((pruned/255).astype(np.uint8))
    
    # width from center
    result = result * 2
    
    # keep the nonzero values
    result = result[np.nonzero(result)]
    
    # if the result is null
    if result.size == 0:
        return 0
    result = np.round(result, 2)
    
    return result


def crack_dist(img_path):
    # image read
    im = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    pix = im/255
    pix = np.where(pix>0.5,255,0)
    pix = pix.astype(np.uint8)

    # skeleton
    skeleton = pcv.morphology.skeletonize(mask=pix)

    # prune
    pruned, seg_img, edge_objects = pcv.morphology.prune(skel_img=skeleton, size=100)

    # distance transform
    dist_transform = cv.distanceTransform(pix, cv.DIST_L2, 5)
    
    # points selection
    result = dist_transform * ((pruned/255).astype(np.uint8))
    
    # width from center
    result = result * 2
    
    # keep the nonzero values
    result = result[np.nonzero(result)]

    # if the result is null
    if result.size == 0:
        return 0
    result = np.round(result, 2)
    
    plt.figure()
    sns.distplot(result)
    plt.show()
    
def categorical_crack(value):
    if value < 10:
        return 'grade_1'
    elif 10 <= value < 20:
        return 'grade_2'
    elif 20 <= value < 30:
        return 'grade_3'
    elif 30 <= value < 40:
        return 'grade_4'
    elif 40 <= value < 50:
        return 'grade_5'
    elif 50 <= value < 60:
        return 'grade_6'
    elif 60 <= value < 70:
        return 'grade_7'
    elif 70 <= value < 80:
        return 'grade_8'
    elif 80 <= value < 90:
        return 'grade_9'
    elif 90 <= value < 100:
        return 'grade_10'
    else:
        return 'spalling'

# image show
def img_show(img_path):
    im = pilimg.open(img_path)
    pix = np.array(im)
    plt.figure()
    plt.axis('off')
    plt.imshow(pix)