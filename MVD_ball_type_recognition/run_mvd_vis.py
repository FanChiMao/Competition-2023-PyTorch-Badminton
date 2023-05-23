# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 09:06:22 2023

@author: ms024
"""

import os
import cv2
import csv
import time
import glob
import argparse
from timm.models import create_model
import modeling_finetune
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from decord import VideoReader, cpu
from torchvision import transforms
from transforms import Stack, GroupNormalize, GroupMultiScaleTwoResizedCrop, ToTorchFormatTensor
from masking_generator import  TubeMaskingGenerator, RandomMaskingGenerator
import utils
from ultralytics import YOLO

def pad_and_resize(img):
    h, w, _ = img.shape
    max_size = max(h, w)
    if (max_size - h) > 0:
        top = bottom = (max_size - h)//2
    else:
        top = bottom = 0
    
    if (max_size - w) > 0:
        left = right = (max_size - w)//2
    else:
        left = right = 0
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img

def get_player_roi(detect_players, img):
    result_player = detect_players.predict(source=img, save=False, imgsz=640, max_det=2, device='cpu')
    try:
        player1 = result_player[0][0]
        player2 = result_player[0][1]
        
        class_id = player1.boxes.cls.item()
        xyxy_1 = list(player1.boxes.xyxy.cpu().numpy()[0])
        xyxy_2 = list(player2.boxes.xyxy.cpu().numpy()[0])
        xyxy_1 = [int(j) for j in xyxy_1]
        xyxy_2 = [int(j) for j in xyxy_2]
        xyxy_A, xyxy_B = [xyxy_1, xyxy_2] if class_id == 1 else [xyxy_2, xyxy_1]
    except:
        print('Our model only detect 1 player...')
        xyxy_A, xyxy_B = None, None
    
    return xyxy_A, xyxy_B

class DataAugmentationForVideoDistillation(object):
    def __init__(self, args, num_frames=None):
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleTwoResizedCrop(
            args.input_size, args.input_size, [1, .875, .75, .66]
        )
        self.transform = transforms.Compose([
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        window_size = args.window_size if num_frames is None else (num_frames // args.tubelet_size, args.window_size[1], args.window_size[2])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                window_size, args.mask_ratio
            )
        elif args.mask_type == 'random':
            self.masked_position_generator = RandomMaskingGenerator(
                window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data_0, process_data_1, labels = self.train_augmentation(images)
        process_data_0, _ = self.transform((process_data_0, labels))
        process_data_1, _ = self.transform((process_data_1, labels))
        return process_data_0, process_data_1, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoDistillation,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def get_args():
    parser = argparse.ArgumentParser('MVD fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--img_path', type=str, help='input video path')
    parser.add_argument('--save_path', type=str, help='save video path')
    parser.add_argument('--model_path', type=str, help='checkpoint path of model')
    parser.add_argument('--mask_type', default='random', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--nb_classes', default=10, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_cls_token', action='store_true', default=False)
    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--dist_on_itp', action='store_true')

    return parser.parse_args()

def AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if(isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontText)
    
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def run_one_video(args, vid_path, detect_players, device, model):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(os.path.join(args.save_path, os.path.split(vid_path)[-1]), fourcc, fps, (width,  height))
    ball_type = ["無", "發短球", "發長球", "長球", "平球",
                 "殺球", "網前小球", "切球", "挑球", "推撲球"]
    
    with open(vid_path, 'rb') as f:
        vr = VideoReader(f, ctx=cpu(0))
    duration = len(vr)
    
    transforms = DataAugmentationForVideoDistillation(args, args.num_frames)
    
    frame_id_list = []
    img_A = []
    img_B = []
    vis_data = []
    csvfile_write = open(os.path.join(args.save_path, os.path.split(vid_path)[-1].replace(".mp4", ".csv")), 'w', newline='')
    writer = csv.writer(csvfile_write)
    
    for i in range(duration):
        frame_id_list.append(i)
        if len(frame_id_list) < args.num_frames:
            video_data = vr.get_batch([i]).asnumpy()
            img_less_16 = Image.fromarray(video_data[0, :, :, :]).convert('RGB')
            out_frame_less_16 = cv2.cvtColor(np.array(img_less_16), cv2.COLOR_RGB2BGR)
            
            xyxy_A, xyxy_B = get_player_roi(detect_players, out_frame_less_16)
            if xyxy_A == None or xyxy_B == None:
                continue
            
            if xyxy_A[1] >= xyxy_B[1]: ### upper A, bottom B
                img_player_B = out_frame_less_16[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
                img_player_A = out_frame_less_16[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
                vis_data.append([out_frame_less_16, xyxy_B, xyxy_A])
                
            else:
                img_player_A = out_frame_less_16[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
                img_player_B = out_frame_less_16[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
                vis_data.append([out_frame_less_16, xyxy_A, xyxy_B])
            
            img_player_A = pad_and_resize(img_player_A)
            img_player_B = pad_and_resize(img_player_B)
            img_A.append(Image.fromarray(cv2.cvtColor(img_player_A, cv2.COLOR_BGR2RGB)))
            img_B.append(Image.fromarray(cv2.cvtColor(img_player_B, cv2.COLOR_BGR2RGB)))
            
            if i >= (args.num_frames/2):
                print(f"i = {i}, i-(args.num_frames/2)-1 = {i-7}")
                vis_frame = vis_data[i-8][0]
                vis_xyxy_A = vis_data[i-8][1]
                vis_xyxy_B = vis_data[i-8][2]
                cv2.rectangle(vis_frame, (vis_xyxy_A[0], vis_xyxy_A[1]), (vis_xyxy_A[2], vis_xyxy_A[3]), (0, 0, 255), 1)
                cv2.rectangle(vis_frame, (vis_xyxy_B[0], vis_xyxy_B[1]), (vis_xyxy_B[2], vis_xyxy_B[3]), (0, 0, 255), 1)
            
                out.write(vis_frame)
                # cv2.imshow("out_frame", vis_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
            continue
        elif len(frame_id_list) > args.num_frames:
            frame_id_list.pop(0)
            
        if len(img_A) >= args.num_frames:
            img_A.pop(0)
            img_B.pop(0)
    
        video_data = vr.get_batch(frame_id_list).asnumpy()
        
        img_pil = Image.fromarray(video_data[len(frame_id_list)-1, :, :, :]).convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        start = time.time()
        xyxy_A, xyxy_B = get_player_roi(detect_players, img_cv2)
        
        if xyxy_A == None or xyxy_B == None:
                continue
        
        if xyxy_A[1] >= xyxy_B[1]: ### upper A, bottom B
            img_player_B = img_cv2[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
            img_player_A = img_cv2[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
            
        else:
            img_player_A = img_cv2[xyxy_A[1]:xyxy_A[3], xyxy_A[0]:xyxy_A[2], :]
            img_player_B = img_cv2[xyxy_B[1]:xyxy_B[3], xyxy_B[0]:xyxy_B[2], :]
        
        img_player_A = pad_and_resize(img_player_A)
        img_player_B = pad_and_resize(img_player_B)
        img_A.append(Image.fromarray(cv2.cvtColor(img_player_A, cv2.COLOR_BGR2RGB)))
        img_B.append(Image.fromarray(cv2.cvtColor(img_player_B, cv2.COLOR_BGR2RGB)))

        img_A_tensor, _, _ = transforms((img_A, None)) # T*C,H,W
        img_A_tensor = img_A_tensor.view((args.num_frames , 3) + img_A_tensor.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        img_B_tensor, _, _ = transforms((img_B, None)) # T*C,H,W
        img_B_tensor = img_B_tensor.view((args.num_frames , 3) + img_B_tensor.size()[-2:]).transpose(0,1) # T*C,H,W -> T,C,H,W -> C,T,H,W
    
        with torch.no_grad():
            img_A_tensor = img_A_tensor.unsqueeze(0)
            img_A_tensor = img_A_tensor.to(device, non_blocking=True)
            img_B_tensor = img_B_tensor.unsqueeze(0)           
            img_B_tensor = img_B_tensor.to(device, non_blocking=True)
            
            outputs_A = model(img_A_tensor)
            outputs_B = model(img_B_tensor)
        
        end = time.time()
        print(f"frame = {i-8}, outputs_A = {outputs_A.argmax()}, outputs_B = {outputs_B.argmax()}, time = {end - start:>.6f}, len(vis_data)-8 = {len(vis_data)-8}")
        
        if xyxy_A[1] >= xyxy_B[1]: ### upper A, bottom B
            vis_data.append([img_cv2, xyxy_B, xyxy_A])
            
        else:
            vis_data.append([img_cv2, xyxy_A, xyxy_B])
        
        out_frame = vis_data[len(vis_data)-8][0]
        vis_xyxy_A = vis_data[len(vis_data)-8][1]
        vis_xyxy_B = vis_data[len(vis_data)-8][2]
        cv2.rectangle(out_frame, (vis_xyxy_A[0], vis_xyxy_A[1]), (vis_xyxy_A[2], vis_xyxy_A[3]), (0, 0, 255), 3)
        cv2.rectangle(out_frame, (vis_xyxy_B[0], vis_xyxy_B[1]), (vis_xyxy_B[2], vis_xyxy_B[3]), (0, 0, 255), 3)
        
        type_text_A = ball_type[outputs_A.argmax()]
        type_text_B = ball_type[outputs_B.argmax()]
        writer.writerow([i-8, outputs_A.argmax().detach().cpu().numpy(), outputs_B.argmax().detach().cpu().numpy()])
        out_frame = AddChineseText(out_frame, type_text_A, (vis_xyxy_A[0], vis_xyxy_A[1]-25), textColor=(255, 0, 0), textSize=20)
        out_frame = AddChineseText(out_frame, type_text_B, (vis_xyxy_B[0], vis_xyxy_B[1]-25), textColor=(255, 0, 0), textSize=20)
        
        # cv2.imshow("out_frame", out_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        out.write(out_frame)
    out.release()
    csvfile_write.close()

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)
    cudnn.benchmark = True    
    
    detect_players = YOLO(r"E:\Job\ASUS\AICUP\Competition-2023-PyTorch-Badminton\trained_weights\yolov8s_players_detection_2.pt")
    
    model = create_model(
        args.model,
        pretrained=False,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        use_cls_token=args.use_cls_token,
        fc_drop_rate=args.fc_drop_rate,
        use_checkpoint=args.use_checkpoint,
    )
    
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    if args.save_path:
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
    
    vid_dirs = os.listdir(args.img_path)
    for vid_dir in vid_dirs:
        if int(vid_dir) < 171:
            continue
        vid_path_tmp = os.path.join(args.img_path, vid_dir)
        vid_path = glob.glob(os.path.join(vid_path_tmp, "*.mp4"))[0]
    
        run_one_video(args, vid_path, detect_players, device, model)

if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)