import sys
import os
import numpy as np
from pathlib import Path
import cv2 as cv
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from unet.unet_transfer import UNet16, input_size
import matplotlib.pyplot as plt
from matplotlib.image import imread
import argparse
from os.path import join
from PIL import Image
import gc
from utils import load_unet_vgg16, load_unet_vgg16_bn, load_unet_vgg16_bn_do, load_unet_vgg16_fullbn_do, load_unet_3plus, load_unet_2plus, load_unet_resnet_101, load_unet_resnet_34
from tqdm import tqdm

def evaluate_img(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_1 = cv.resize(img, (input_width, input_height), cv.INTER_AREA)
    X = train_tfms(Image.fromarray(img_1))
    X = Variable(X.unsqueeze(0)).cuda()  # [N, 1, H, W]
    
    mask = model(X)

    mask = F.sigmoid(mask[0, 0]).data.cpu().numpy()
    mask = cv.resize(mask, (img_width, img_height), cv.INTER_AREA)
    return mask

def evaluate_img_patch(model, img):
    input_width, input_height = input_size[0], input_size[1]

    img_height, img_width, img_channels = img.shape

    if img_width < input_width or img_height < input_height:
        return evaluate_img(model, img)

    stride_ratio = 0.1
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []
    for y in range(0, img_height - input_height + 1, stride):
        for x in range(0, img_width - input_width + 1, stride):
            segment = img[y:y + input_height, x:x + input_width]
            normalization_map[y:y + input_height, x:x + input_width] += 1
            patches.append(segment)
            patch_locs.append((x, y))

    patches = np.array(patches)
    if len(patch_locs) <= 0:
        return None

    preds = []
    for i, patch in enumerate(patches):
        patch_n = train_tfms(Image.fromarray(patch))
        X = Variable(patch_n.unsqueeze(0)).cuda()  # [N, 1, H, W]
        masks_pred = model(X)
        mask = F.sigmoid(masks_pred[0, 0]).data.cpu().numpy()
        preds.append(mask)

    probability_map = np.zeros((img_height, img_width), dtype=float)
    for i, response in enumerate(preds):
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response

    return probability_map

def disable_axis():
    plt.axis('off')
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_dir',type=str, help='input dataset directory')
    parser.add_argument('-model_path', type=str, help='trained model path')
    parser.add_argument('-model_type', type=str)
    parser.add_argument('-out_viz_dir', type=str, default='', required=False, help='visualization output dir')
    parser.add_argument('-out_pred_dir', type=str, default='', required=False,  help='prediction output dir')
    parser.add_argument('-threshold', type=float, default=0.5 , help='threshold to cut off crack response')
    args = parser.parse_args()

    if args.out_viz_dir != '':
        os.makedirs(args.out_viz_dir, exist_ok=True)
        for path in Path(args.out_viz_dir).glob('*.*'):
            os.remove(str(path))

    if args.out_pred_dir != '':
        os.makedirs(args.out_pred_dir, exist_ok=True)
        for path in Path(args.out_pred_dir).glob('*.*'):
            os.remove(str(path))

    if args.model_type == 'vgg16':
        model = load_unet_vgg16(args.model_path)
    elif args.model_type == 'vgg16_bn':
        model = load_unet_vgg16_bn(args.model_path)
    elif args.model_type == 'vgg16_bn_do':
        model = load_unet_vgg16_bn_do(args.model_path)
    elif args.model_type == 'vgg16_fullbn_do':
        model = load_unet_vgg16_fullbn_do(args.model_path)
    elif args.model_type == 'unet+++':
        model = load_unet_3plus(args.model_path)
    elif args.model_type == 'unet++':
        model = load_unet_2plus(args.model_path)
    elif args.model_type  == 'resnet101':
        model = load_unet_resnet_101(args.model_path)
    elif args.model_type  == 'resnet34':
        model = load_unet_resnet_34(args.model_path)
    else:
        print('undefind model name pattern')
        exit()

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    
#     test_mask_path = '/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/masks/'
#     test_image_path = '/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/test/images/'
    
    test_image_path = args.img_dir
    test_mask_path = os.path.join(os.path.split(test_image_path)[0], 'masks')
    
    paths = [path for path in Path(args.img_dir).glob('*.*')]
    for path in tqdm(paths):
        #print(str(path))

        train_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(channel_means, channel_stds)])
        
        # have to find to delete .ipynb_checkpoints
        if 'ipynb_checkpoints' in str(path):
            continue
        img_0 = Image.open(str(path))
        img_0 = np.asarray(img_0)
        if len(img_0.shape) != 3:
            print(f'incorrect image shape: {path.name}{img_0.shape}')
            continue

        img_0 = img_0[:,:,:3]

        img_height, img_width, img_channels = img_0.shape

        prob_map_full = evaluate_img(model, img_0)

        if args.out_pred_dir != '':
            prob_map_full_tmp = np.where(prob_map_full < args.threshold, 0, 1)    # segmentation method
            cv.imwrite(filename=join(args.out_pred_dir, f'{path.stem}.jpg'), img=(prob_map_full_tmp * 255).astype(np.uint8))

        if args.out_viz_dir != '':
            # plt.subplot(121)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            if img_0.shape[0] > 2000 or img_0.shape[1] > 2000:
                img_1 = cv.resize(img_0, None, fx=0.2, fy=0.2, interpolation=cv.INTER_AREA)
            else:
                img_1 = img_0

            # plt.subplot(122)
            # plt.imshow(img_0), plt.title(f'{img_0.shape}')
            # plt.show()

            prob_map_patch = evaluate_img_patch(model, img_1)

            #plt.title(f'name={path.stem}. \n cut-off threshold = {args.threshold}', fontsize=4)
            
            prob_map_viz_patch = prob_map_patch.copy()
            
            # repo method
#             prob_map_viz_patch = prob_map_viz_patch/ prob_map_viz_patch.max()
            
            # segmentation method
            prob_map_viz_patch = np.where(prob_map_viz_patch<args.threshold, 0, 1)
            fig = plt.figure()
            st = fig.suptitle(f'name={path.stem} \n cut-off threshold = {args.threshold}', fontsize="large")
            ax1 = fig.add_subplot(231)
            ax1.title.set_text('Original Image')
            ax1.imshow(img_1)
            ax2 = fig.add_subplot(232)
            ax2.title.set_text('Predicted Mask')
            ax2.imshow(prob_map_viz_patch)
            ax3 = fig.add_subplot(233)
            ax3.title.set_text('Predicted Overlap')
            ax3.imshow(img_1)
            ax3.imshow(prob_map_viz_patch, alpha=0.4)

#             prob_map_viz_full = prob_map_full.copy()
#             prob_map_viz_full[prob_map_viz_full < args.threshold] = 0.0

#             ax = fig.add_subplot(234)
#             ax.imshow(img_0)
#             ax = fig.add_subplot(235)
#             ax.imshow(prob_map_viz_full)
#             ax = fig.add_subplot(236)
#             ax.imshow(img_0)
#             ax.imshow(prob_map_viz_full, alpha=0.4)


            # true image info
            test_mask = imread(os.path.join(test_mask_path, path.name))
            test_image = imread(os.path.join(test_image_path, path.name))
        
            ax4 = fig.add_subplot(234)
            ax4.title.set_text('Original Image')
            ax4.imshow(test_image)
            ax5 = fig.add_subplot(235)
            ax5.title.set_text('True Mask')
            ax5.imshow(test_mask)
            ax6 = fig.add_subplot(236)
            ax6.title.set_text('True Overlap')
            ax6.imshow(test_image)
            ax6.imshow(test_mask, alpha=0.4)

            plt.savefig(join(args.out_viz_dir, f'{path.stem}.jpg'), dpi=500)
            plt.close('all')

        gc.collect()