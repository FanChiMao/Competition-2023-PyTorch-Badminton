import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger(log_dir: str, log_file: str) -> None:
    logger = logging.getLogger()
    format_str = r'[%(asctime)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        datefmt=r'%Y/%m/%d %H:%M:%S',
        format=format_str
    )
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(str(log_dir / log_file))
    fh.setFormatter(logging.Formatter(format_str))
    logger.addHandler(fh)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # common config
    parser.add_argument('--model', type=str, choices=('anchor-free'), default='anchor-free')
    parser.add_argument('--base-model', type=str, default='attention', choices=['attention', 'lstm', 'linear', 'bilstm', 'gcn'])
    parser.add_argument('--num-head', type=int, default=8)
    parser.add_argument('--num-feature', type=int, default=1000)
    parser.add_argument('--num-hidden', type=int, default=128)
    parser.add_argument('--model-dir', type=str, default='../models')
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--device', type=str, default='cuda', choices=('cuda', 'cpu'))

    # training
    parser.add_argument('--max-epoch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--cls-loss', type=str, default='focal', choices=['focal','cross-entropy'])

    # training & evaluation
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--splits', type=str, nargs='+', default=["../custom_data/yml/badminton_clean_sub.yml"])

    # evaluation & inference
    parser.add_argument('--ckpt-path', type=str, default='../models/checkpoint/badminton_clean_sub.yml.0.pt')
    parser.add_argument('--save-dir', type=str, default='../output_fig/test_videos')

    # inference
    parser.add_argument('--test-source', type=str, default='D:/Dataset/AICUP/part1/val')
    parser.add_argument('--sample-rate', type=int, default=1)
    parser.add_argument('--backbone', type=str, choices=('swin_v2_t', 'regnet_y_16gf', 'googlenet'), default='swin_v2_t')

    return parser


def get_arguments() -> argparse.Namespace:
    parser = get_parser()
    args = parser.parse_args()
    return args
