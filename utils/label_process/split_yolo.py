import os
from sklearn.model_selection import train_test_split
import argparse
import glob
import shutil


def random_split(folders, val_size):
    image_data = glob.glob(os.path.join(folders, '*.png'))
    label_data = glob.glob(os.path.join(folders, '*.txt'))

    assert len(image_data) == len(label_data)
    train_idxs, val_idxs = train_test_split(range(len(image_data)), test_size=val_size)

    train_names = [image_data[train_idx] for train_idx in train_idxs]
    val_names = [image_data[val_idx] for val_idx in val_idxs]

    return train_names, val_names





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r"D:\AICUP\Badminton\dataset\Public\labels_for_players",
                        help='The path includes both images and yolo txt labels.')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Please input the validation dataset size, for example 0.1 ')
    parser.add_argument('--save_dir', type=str, default="./",
                        help='Please input the validation dataset size, for example 0.1 ')
    args = parser.parse_args()

    save_root = os.path.join(args.save_dir, "YOLODataset")
    os.makedirs(save_root, exist_ok=True)

    save_path_images = os.path.join(save_root, "images")
    save_path_labels = os.path.join(save_root, "labels")
    os.makedirs(save_path_images, exist_ok=True)
    os.makedirs(save_path_labels, exist_ok=True)

    save_path_images_train = os.path.join(save_path_images, "train")
    save_path_images_valid = os.path.join(save_path_images, "val")
    os.makedirs(save_path_images_train, exist_ok=True)
    os.makedirs(save_path_images_valid, exist_ok=True)

    save_path_labels_train = os.path.join(save_path_labels, "train")
    save_path_labels_valid = os.path.join(save_path_labels, "val")
    os.makedirs(save_path_labels_train, exist_ok=True)
    os.makedirs(save_path_labels_valid, exist_ok=True)

    train_set, valid_set = random_split(args.data_dir, args.val_size)

    for train_image in train_set:
        label_name = train_image.replace('.png', '.txt')
        shutil.copyfile(train_image, os.path.join(save_path_images_train, os.path.basename(train_image)))
        shutil.copyfile(label_name, os.path.join(save_path_labels_train, os.path.basename(label_name)))

    for train_image in valid_set:
        label_name = train_image.replace('.png', '.txt')
        shutil.copyfile(train_image, os.path.join(save_path_images_valid, os.path.basename(train_image)))
        shutil.copyfile(label_name, os.path.join(save_path_labels_valid, os.path.basename(label_name)))
    pass

