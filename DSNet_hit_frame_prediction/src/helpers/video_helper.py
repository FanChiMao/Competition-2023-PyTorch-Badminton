from os import PathLike
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from numpy import linalg
from torch import nn
from torchvision import transforms, models
from torchvision.models import swin_v2_t, Swin_V2_T_Weights, regnet_y_16gf, RegNet_Y_16GF_Weights, regnet_y_128gf, RegNet_Y_128GF_Weights, googlenet, GoogLeNet_Weights


class FeatureExtractor(object):
    def __init__(self, backbone: str):
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self.bulid_feature_extractor(backbone).cuda().eval()

    def run(self, img: np.ndarray) -> np.ndarray:
        img = Image.fromarray(img)
        img = self.preprocess(img)
        batch = img.unsqueeze(0)
        with torch.no_grad():
            feat = self.model(batch.cuda())
            feat = feat.squeeze().cpu().numpy()

        feat /= linalg.norm(feat) + 1e-10
        return feat
    
    def bulid_feature_extractor(self, backbone: str):
        if backbone == 'swin_v2_t':
            weights = Swin_V2_T_Weights.DEFAULT
            model = swin_v2_t(weights=weights)
            #TODO
        elif backbone == 'regnet_y_16gf':
            weights = RegNet_Y_16GF_Weights.DEFAULT
            model = regnet_y_16gf(weights=weights)
            #TODO
        elif backbone == 'googlenet':
            model = models.googlenet(pretrained=True)
            model = nn.Sequential(*list(model.children())[:-2])
        else:
            raise ValueError(f'Invalid backbone {backbone}')
        
        return model


class VideoPreprocessor(object):
    def __init__(self, sample_rate: int, backbone: str) -> None:
        self.model = FeatureExtractor(backbone)
        self.sample_rate = sample_rate

    def get_features(self, video_path: PathLike):
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap is not None, f'Cannot open video: {video_path}'

        features = []
        n_frames = 0

        fps = cap.get(cv2.CAP_PROP_FPS)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if n_frames % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feat = self.model.run(frame)
                features.append(feat)

            n_frames += 1

        cap.release()

        features = np.array(features)
        return n_frames, features

    def kts(self, n_frames, features):
        change_points = []
        n_frame_per_seg = []
        picks = []
        return change_points, n_frame_per_seg, picks

    def run(self, video_path: PathLike):
        n_frames, features = self.get_features(video_path)
        cps, nfps, picks = self.kts(n_frames, features)
        return n_frames, features, cps, nfps, picks
