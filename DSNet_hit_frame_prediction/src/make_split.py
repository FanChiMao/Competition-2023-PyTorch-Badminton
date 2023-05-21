import argparse
import random
from pathlib import Path

import h5py
import yaml


def make_random_splits(keys, num_test, num_splits):
    splits = []
    for _ in range(num_splits):
        random.shuffle(keys)
        test_keys = keys[:num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def make_cross_val_splits(keys, num_videos, num_test):
    random.shuffle(keys)
    splits = []
    for i in range(0, num_videos, num_test):
        test_keys = keys[i: i + num_test]
        train_keys = list(set(keys) - set(test_keys))
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys
        })
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../custom_data/h5/badminton_clean_sub.h5',
                        help='Path to h5 dataset')
    parser.add_argument('--save-path', type=str, default='../custom_data/yml/badminton_clean_sub.yml',
                        help='Path to save generated splits')
    parser.add_argument('--num-splits', type=int, default=1,
                        help='How many splits to generate')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Percentage of training data')
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'cross'],
                        help='Random selection or cross validation')
    args = parser.parse_args()

    dataset = h5py.File(args.dataset, 'r')
    keys = list(dataset.keys())
    keys = [str(Path(args.dataset) / key) for key in keys]

    num_videos = len(keys)
    num_train = round(num_videos * args.train_ratio)
    num_test = num_videos - num_train

    if args.method == 'random':
        splits = make_random_splits(keys, num_test, args.num_splits)
    elif args.method == 'cross':
        splits = make_cross_val_splits(keys, num_videos, num_test)
    else:
        raise ValueError(f'Invalid method {args.method}')

    # save splits
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(splits, f)


if __name__ == '__main__':
    main()
