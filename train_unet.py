import torch
from torch import nn
from unet.unet_transfer import UNet16, UNet16_bn, UNet16_bn_do, UNet16_fullbn_do, UNetResNet
from unet.UNet_3Plus import UNet_3Plus
from unet.UNet_2Plus import UNet_2Plus
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from torch.autograd import Variable
import shutil
from data_loader import ImgDataSet, ImgDataSetJoint
import os
import argparse
import tqdm
import numpy as np
import scipy.ndimage as ndimage
from pytorchtools import EarlyStopping
from lossfunction import *
from aug_imgaug import ImgAugTransform
import random


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_model(device, type ='vgg16'):
    if type == 'vgg16':
        print('create vgg16 model')
        model = UNet16(pretrained=True)
    elif type == 'vgg16_bn':
        print('create vgg16_bn model')
        model = UNet16_bn(pretrained=True)
    elif type == 'vgg16_bn_do':
        print('create vgg16_bn_do model')
        model = UNet16_bn_do(pretrained=True)
    elif type == 'vgg16_fullbn_do':
        print('create vgg16_fullbn_do model')
        model = UNet16_fullbn_do(pretrained=True)
    elif type == 'resnet101':
        encoder_depth = 101
        num_classes = 1
        print('create resnet101 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'resnet34':
        encoder_depth = 34
        num_classes = 1
        print('create resnet34 model')
        model = UNetResNet(encoder_depth=encoder_depth, num_classes=num_classes, pretrained=True)
    elif type == 'unet+++':
        model = UNet_3Plus()
        print('create U-Net+++ model')
    elif type =='unet++':
        model = UNet_2Plus()
        print('create U-Net++ model')
    else:
        assert False
    model.eval()
    return model.to(device)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def find_latest_model_path(dir):
    model_paths = []
    epochs = []
    for path in Path(dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        model_paths.append(path)
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        epochs.append(epoch)

    if len(epochs) > 0:
        epochs = np.array(epochs)
        max_idx = np.argmax(epochs)
        return model_paths[max_idx]
    else:
        return None

def train(train_loader, model, criterion, optimizer, validation, args):

    latest_model_path = find_latest_model_path(args.model_dir)

    best_model_path = os.path.join(*[args.model_dir, 'model_best.pt'])

    # Early Stopping
    early_stopping = EarlyStopping(patience=7, verbose=False)
    
    if latest_model_path is not None:
        state = torch.load(latest_model_path)
        epoch = state['epoch']
        model.load_state_dict(state['model'])
        epoch = epoch

        #if latest model path does exist, best_model_path should exists as well
        assert Path(best_model_path).exists() == True, f'best model path {best_model_path} does not exist'
        #load the min loss so far
        best_state = torch.load(latest_model_path)
        min_val_los = best_state['valid_loss']

        print(f'Restored model at epoch {epoch}. Min validation loss so far is : {min_val_los}')
        epoch += 1
        print(f'Started training model from epoch {epoch}')
    else:
        print('Started training model from epoch 0')
        epoch = 0
        min_val_los = 9999

    valid_losses = []
    for epoch in range(epoch, args.n_epoch + 1):

        adjust_learning_rate(optimizer, epoch, args.lr)

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description(f'Epoch {epoch}')

        losses = AverageMeter()

        model.train()
        for i, (input, target) in enumerate(train_loader):
            input_var  = Variable(input).cuda()
            target_var = Variable(target).cuda()

            masks_pred = model(input_var)

            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat  = target_var.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            losses.update(loss)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg))
            tq.update(args.batch_size)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_metrics = validation(model, valid_loader, criterion)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        print(f'\tvalid_loss = {valid_loss:.5f}')
        tq.close()

        #save the model of the current epoch
        epoch_model_path = os.path.join(*[args.model_dir, f'model_epoch_{epoch}.pt'])
        torch.save({
            'model': model.state_dict(),
            'epoch': epoch,
            'valid_loss': valid_loss,
            'train_loss': losses.avg
        }, epoch_model_path)

        if valid_loss < min_val_los:
            min_val_los = valid_loss

            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'valid_loss': valid_loss,
                'train_loss': losses.avg
            }, best_model_path)
            
        # Early Stopping
        early_stopping(valid_loss)
        
        if early_stopping.early_stop:
            print('Early stopping')
            break

def validate(model, val_loader, criterion):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad():

        for i, (input, target) in enumerate(val_loader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()

            output = model(input_var)
            loss = criterion(output, target_var)

            losses.update(loss.item(), input_var.size(0))

    return {'valid_loss': losses.avg}

def save_check_point(state, is_best, file_name = 'checkpoint.pth.tar'):
    torch.save(state, file_name)
    if is_best:
        shutil.copy(file_name, 'model_best.pth.tar')

def calc_crack_pixel_weight(mask_dir):
    avg_w = 0.0
    n_files = 0
    for path in Path(mask_dir).glob('*.*'):
        n_files += 1
        m = ndimage.imread(path)
#         m = matplotlib.pyplot.imread(path)
        ncrack = np.sum((m > 0)[:])
        w = float(ncrack)/(m.shape[0]*m.shape[1])
        avg_w = avg_w + (1-w)

    avg_w /= float(n_files)

    return avg_w / (1.0 - avg_w)

if __name__ == '__main__':
    
    # fix the randomness
    random_seed=486
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-n_epoch', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-print_freq', default=20, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-batch_size',  default=4, type=int,  help='weight decay (default: 1e-4)')
    parser.add_argument('-num_workers', default=4, type=int, help='output dataset directory')

#     parser.add_argument('-data_dir',type=str, help='input dataset directory')
    parser.add_argument('-train_dir', default='/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train_train', type=str, help='input dataset directory')
    parser.add_argument('-val_dir', default='/home/jovyan/work/hangman/dataset/crack_segmentation_dataset/train_val', type=str, help='input dataset directory')
    parser.add_argument('-model_dir', type=str, help='output dataset directory')
    parser.add_argument('-model_type', type=str, required=False, default='resnet101')
    
    # image augmentation
    parser.add_argument('-augmentation', type=bool, default=False)
    
    # loss function selection
    parser.add_argument('-lossft', type=str, default='BCE')

    args = parser.parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # train and val fitting

    TRAIN_DIR_IMG = os.path.join(args.train_dir, 'images')
    TRAIN_DIR_MASK = os.path.join(args.train_dir, 'masks')
    train_img_names = [path.name for path in Path(TRAIN_DIR_IMG).glob('*.jpg')]
    train_mask_names = [path.name for path in Path(TRAIN_DIR_MASK).glob('*.jpg')]
    
    VAL_DIR_IMG = os.path.join(args.val_dir, 'images')
    VAL_DIR_MASK = os.path.join(args.val_dir, 'masks')
    val_img_names = [path.name for path in Path(VAL_DIR_IMG).glob('*.jpg')]
    val_mask_names = [path.name for path in Path(VAL_DIR_MASK).glob('*.jpg')]

    print(f'total images = {len(train_img_names)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = create_model(device, args.model_type)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    

    
    # loss function selection
    if args.lossft == 'focal':    # focal loss
        criterion = FocalLoss()
    elif args.lossft == 'infogain':    # infogain loss
        DIR_IMG  = os.path.join(args.train_dir, 'images')
        DIR_MASK = os.path.join(args.train_dir, 'masks')

        img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
        mask_names = [path.name for path in Path(DIR_MASK).glob('*.jpg')]

        crack_weight = 0.4*calc_crack_pixel_weight(DIR_MASK)
        print(f'positive weight: {crack_weight}')
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([crack_weight]).to('cuda'))
    elif args.lossft == 'BCE':
        criterion = nn.BCEWithLogitsLoss().to('cuda')
    elif args.lossft == 'dice':
        criterion = DiceLoss()
    elif args.lossft == 'bcedice':
        criterion = DiceBCELoss()
    elif args.lossft == 'logcoshdice':
        criterion = LogCoshDiceLoss()
    elif args.lossft == 'focaltversky':
        criterion = FocalTverskyLoss()
    else:
        assert 0, 'Please set the loss function again'
    

    channel_means = [0.485, 0.456, 0.406]
    channel_stds  = [0.229, 0.224, 0.225]
    
    
    train_tfms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(channel_means, channel_stds)])

    val_tfms = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(channel_means, channel_stds)])

    mask_tfms = transforms.Compose([transforms.ToTensor()])

#     dataset = ImgDataSet(img_dir=DIR_IMG, img_fnames=img_names, img_transform=train_tfms, mask_dir=DIR_MASK, mask_fnames=mask_names, mask_transform=mask_tfms)
#     train_size = int(0.85*len(dataset))
#     valid_size = len(dataset) - train_size
#     train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    
    # Online augmentation is only applied to train dataset
    if args.augmentation:
        train_dataset = ImgDataSetJoint(img_dir=TRAIN_DIR_IMG,
                                   img_fnames=train_img_names,
                                   joint_transform=ImgAugTransform(),
                                   img_transform=train_tfms,
                                   mask_dir=TRAIN_DIR_MASK,
                                   mask_fnames=train_mask_names,
                                   mask_transform=mask_tfms)
        valid_dataset = ImgDataSetJoint(img_dir=VAL_DIR_IMG,
                                   img_fnames=val_img_names,
                                   joint_transform=ImgAugTransform(),
                                   img_transform=val_tfms,
                                   mask_dir=VAL_DIR_MASK,
                                   mask_fnames=val_mask_names,
                                   mask_transform=mask_tfms)
    else:
        train_dataset = ImgDataSet(img_dir=TRAIN_DIR_IMG,
                                   img_fnames=train_img_names,
                                   img_transform=train_tfms,
                                   mask_dir=TRAIN_DIR_MASK,
                                   mask_fnames=train_mask_names,
                                   mask_transform=mask_tfms)
        valid_dataset = ImgDataSet(img_dir=VAL_DIR_IMG, img_fnames=val_img_names, img_transform=val_tfms, mask_dir=VAL_DIR_MASK, mask_fnames=val_mask_names, mask_transform=mask_tfms)
    


    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=args.num_workers)

    model.cuda()

    train(train_loader, model, criterion, optimizer, validate, args)