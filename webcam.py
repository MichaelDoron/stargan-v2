import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from os.path import join as ospj
from munch import Munch
from torch.backends import cudnn
import torch
from torch import nn
import torchvision
import torchvision.utils as vutils
from core.utils import get_alphas

import core.utils as utils
from core.data_loader import InputFetcher
from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.model import Generator, MappingNetwork, StyleEncoder, FAN
from core.solver import Solver
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
import time
from vision.ssd.config.fd_config import define_img_size
import imageio
import pyfakewebcam
from glitch_this import ImageGlitcher


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def remove_module_from_state_dict(d):
    new_dict = {}
    for k,v in d.items():
        new_dict['module.' + k] = v
        # new_dict[k.replace('module.','')] = v
    return new_dict

def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    define_img_size(480)
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    test_device = 'cuda:0'
    class_names = ['BACKGROUND', 'face']
    model_path = "models/pretrained/version-RFB-320.pth"
    # net_ema_model_path = "/home/michaeldoron/makers/stargan-v2/expr/checkpoints/100000_nets_ema.ckpt"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    net.load(model_path)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=1500, device=test_device)

    glitcher = ImageGlitcher()
    solver = Solver(args)

    loaders = Munch(src=get_test_loader(root=args.src_dir,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers),
                    ref=get_test_loader(root=args.ref_dir,
                                        img_size=args.img_size,
                                        batch_size=args.val_batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers))

    args = solver.args
    nets_ema = solver.nets_ema
    os.makedirs(args.result_dir, exist_ok=True)
    solver._load_checkpoint(args.resume_iter)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    start = time.time()
    current_style = 0
    tf = torchvision.transforms.ToTensor()
    old_middle = [np.inf,np.inf]
    old_coords = [0,0,0,0]
    x_src = Image.fromarray(frame)
    x_src = x_src.resize((256,256))
    x_src = tf(x_src).unsqueeze(0).cuda()
    x_src = (x_src - 0.5) * 2

    frames = []

    image_index = 0
    image_padding = 100
    box_padding = 0.65
    single_image_size = int(256 * 1.5)
    zoom = True
    brady_bunch = False
    interpolation = True
    regular = True
    mix = False
    w1 = np.random.randint(100,200)
    w2 = np.random.randint(w1,100)
    h1 = np.random.randint(100,200)
    h2 = np.random.randint(h1,100)
    loop = False
    glitch = False
    if brady_bunch:
        num_rows = 3
        num_cols = 3
        num_images = num_rows * num_cols - 1
    elif interpolation:
        num_rows = 1
        num_cols = 1
        num_images = 50
        interpolation_steps = 20
    else:
        num_rows = 1
        num_cols = 1
        num_images = 1

    if zoom:
        zoom_width = 712
        zoom_height = 712
        aug_frame = pyfakewebcam.FakeWebcam('/dev/video7', zoom_width, zoom_height)

    fixed_image = Image.open('/slow/data/CelebA-HQ-img/505.jpg')
    fixed_image = fixed_image.resize((256,256))
    fixed_image = tf(fixed_image).unsqueeze(0).cuda()
    fixed_image = (fixed_image - 0.5) * 2
    df = pd.read_csv('/home/michaeldoron/makers/stargan-v2/CelebAMask-HQ-attribute-anno.txt', delimiter=' ')
    conditions = ((df.Male == 1) & (df.Blurry == -1))
    df = df[conditions]
    df = df.sample(num_images)
    ref_images = []
    domains = []
    original_images = []
    original_domains = torch.Tensor(((df.Male.to_numpy() + 1) / 2)).long()
    for i in df.file_name:
        x_ref = Image.open(f'/slow/data/CelebA-HQ-img/{i}')
        x_ref = x_ref.resize((256,256))
        x_ref = tf(x_ref).unsqueeze(0).cuda()
        x_ref = (x_ref - 0.5) * 2
        original_images.append(x_ref)

    if interpolation:
        alphas =  np.linspace(0, 1, interpolation_steps)
        for i in range(num_images - 1):
            start_image = original_images[i]
            end_image = original_images[i + 1]
            for alpha_ind in range(len(alphas)):
                alpha = alphas[alpha_ind]
                img = torch.lerp(start_image, end_image, alpha)
                ref_images.append(img)
                domains.append(original_domains[i].item())
        start_image = original_images[-1]
        end_image = original_images[0]
        for alpha in alphas:
            ref_images.append(torch.lerp(start_image, end_image, alpha))
            domains.append(original_domains[i].item())
        domains = torch.Tensor(domains).long()
    else:
        ref_images = original_images
        domains = original_domains


    start = time.time()
    while rval:
        augmented_images = []
        with torch.no_grad():
            y_ref = torch.Tensor([1]).long()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, labels, probs = predictor.predict(image, 1500 / 2, 0.6)
            if (len(boxes) > 0):
                box = boxes[0]
                x_start = int(box[1])
                x_end = int(box[3])
                y_start = int(box[0])
                y_end = int(box[2])
                x_range = (x_end - x_start)
                y_range = (y_end - y_start)
                x_middle = (x_start + (x_end - x_start) / 2)
                y_middle = (y_start + (y_end - y_start) / 2)
                x_left = max(0, int(x_middle - x_range * (box_padding + 0.08)))
                x_right = min(frame.shape[0], int(x_middle + x_range * (box_padding - 0.08)))
                y_left = max(0, int(y_middle - x_range * box_padding))
                y_right = min(frame.shape[0], int(y_middle + x_range * box_padding))
                if (np.abs(x_middle - old_middle[0]) / frame.shape[0] > 0.10) or (np.abs(y_middle - old_middle[1]) / frame.shape[1] > 0.10):
                    old_middle = [x_middle, y_middle]
                    old_coords = [x_left, x_right, y_left, y_right]

                image = image[old_coords[0] : old_coords[1], old_coords[2] : old_coords[3], :]
                x_src = Image.fromarray(image)
                x_src = x_src.resize((256,256))
                x_src = tf(x_src).unsqueeze(0).cuda()
                x_src = (x_src - 0.5) * 2

                if brady_bunch:
                    for i in range(len(ref_images)):
                        x_ref = (ref_images[i % len(ref_images)])
                        nets = nets_ema
                        masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
                        s_ref = nets.style_encoder(x_ref, domains[[i % len(ref_images)]])
                        s_ref_list = s_ref.unsqueeze(1)
                        s_ref = s_ref_list[0]
                        x_fake = nets.generator(x_src, s_ref, masks=masks)
                        Ics = torchvision.transforms.Resize((single_image_size,
                                                            single_image_size))(
                            x_fake.detach().cpu()).numpy()[0].transpose(1,2,0)
                        augmented_images.append(Ics)
                else:
                    modulu_image_index = int(image_index) % len(ref_images)
                    if loop:
                        x_ref = (ref_images[modulu_image_index])
                    else:
                        x_ref = fixed_image
                    nets = nets_ema
                    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
                    s_ref = nets.style_encoder(x_ref, domains[[modulu_image_index]])
                    s_ref_list = s_ref.unsqueeze(1)
                    s_ref = s_ref_list[0]
                    x_fake = nets.generator(x_src, s_ref, masks=masks)
                    Ics = torchvision.transforms.Resize((single_image_size,
                                                        single_image_size))(
                        x_fake.detach().cpu()[0])
                    Ics = Ics.numpy().transpose(1,2,0)
                    augmented_images.append(np.array(Ics))

                image_index += 1

        Ics_src = torchvision.transforms.Resize((single_image_size,
                                                 single_image_size))(
            x_src.detach().cpu()).numpy()[0].transpose(1,2,0)
        new_frame = np.zeros((Ics.shape[0] * num_rows, Ics.shape[1] * num_cols, 3))
        if brady_bunch:
            augmented_images = augmented_images[int(len(augmented_images) / 2):] + [Ics_src] + augmented_images[:int(len(augmented_images) / 2)]
        for i in range(num_rows):
            for j in range(num_cols):
                new_frame[i * Ics.shape[0] : (i + 1) * Ics.shape[0],
                          j * Ics.shape[1] : (j + 1) * Ics.shape[1],
                          :] = augmented_images[(i * num_rows) + j]
        x = new_frame.astype(float)

        if zoom:
            if regular:
                x = Ics_src.astype(float)
            elif mix:
                x[w1:w2,h1:h2,:] = Ics_src.astype(float)[w1:w2,h1:h2,:]

            x -= x.min()
            x /= x.max()
            x *= 255
            x = x.astype(np.uint8)
            if glitch and (np.random.rand() < 0.1):
                x = torch.Tensor(x.transpose(2,0,1))
                x = torchvision.transforms.ToPILImage()(x)
                x = (glitcher.glitch_image(src_img = x,
                                        glitch_amount = 0.11 + np.random.rand() * 9.8,
                                        frames = 1,
                                        step = 1,
                                        color_offset = True,
                                        scan_lines = True))
            x = torchvision.transforms.ToTensor()(x)
            x = x.numpy().transpose(1,2,0)
            x -= x.min()
            x /= x.max()
            x *= 255
            x = x.astype(np.uint8)
            new_width_left = int((zoom_width - x.shape[1]) / 2)
            new_height_up = int((zoom_height - x.shape[0]) / 2)
            new_width_right = zoom_width - (x.shape[1] + new_width_left)
            new_height_down = zoom_height - (x.shape[0] + new_height_up)
            pads = ((new_height_up,new_height_down),(new_width_right,new_width_left))
            # pads = ((new_width,new_width),(new_height,new_height))
            x = np.concatenate([np.pad(x[:,:,0],
                                       pads,
                                       'constant')[:,:,np.newaxis],
                                np.pad(x[:,:,1],
                                       pads,
                                       'constant')[:,:,np.newaxis],
                                np.pad(x[:,:,2],
                                       pads,
                                       'constant')[:,:,np.newaxis]],
                               axis=2)

            aug_frame.schedule_frame(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        cv2.imshow("preview", x)

        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 103: # exit on ESC
            if glitch:
                glitch = False
                print('no glitch')
            else:
                glitch = True
                print('glitch')

        elif key == 114: # exit on ESC
            if regular:
                regular = False
                print('no regular')
            else:
                regular = True
                print('regular')

        elif key == 108: # exit on ESC
            if loop:
                print('no loop')
                loop = False
            else:
                print('loop')
                loop = True

        elif key == 109: # exit on ESC
            if mix:
                print('no mix')
                mix = False
            else:
                print('mix')
                mix = True
                w1 = np.random.randint(0,Ics_src.shape[0])
                w2 = np.random.randint(w1,Ics_src.shape[0])
                h1 = np.random.randint(0,Ics_src.shape[1])
                h2 = np.random.randint(h1,Ics_src.shape[1])
                print((w1,w2,h1,h2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=100000,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--mode', type=str, default='sample',
                        choices=['train', 'sample', 'eval', 'align'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='data/celeba_hq/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='data/celeba_hq/val',
                        help='Directory containing validation images')
    parser.add_argument('--sample_dir', type=str, default='expr/samples',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='expr/eval',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='expr/results',
                        help='Directory for saving generated images and videos')
    parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')
    parser.add_argument('--movie_file', type=str, default='movie.gif',
                        help='Directory containing input source images')

    # face alignment
    parser.add_argument('--wing_path', type=str, default='expr/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='expr/checkpoints/celeba_lm_mean.npz')

    # step size
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=50000)

    args = parser.parse_args()
    main(args)

