import logging
import os

from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.signal import find_peaks
import seaborn as sns
import matplotlib.pyplot as plt
import heapq

import numpy as np
import torch

from helpers import init_helper, data_helper
from modules.model_zoo import get_model


logger = logging.getLogger()

def topk_idx(pred_cls, idx_lst, k):
    topk = heapq.nlargest(k, enumerate(pred_cls[idx_lst]), key=lambda x: x[1])
    result_order_idx = [x[0] for x in sorted(topk, key=lambda x: x[0])]
    result = idx_lst[result_order_idx]
    return result

def evaluate(model, val_loader, device):
    model.eval()

    stats = data_helper.AverageMeter('roc_auc', 'dr_acc')

    with torch.no_grad():
        for test_key, seq, hitframe_gt, direction_gt, cps, n_frames, nfps, picks, user_summary,_ in val_loader:
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_dr = model.predict(seq_torch)

            roc_auc = roc_auc_score(hitframe_gt, pred_cls)

            acc = (pred_dr.argmax(1) == direction_gt).sum() / direction_gt.size

            stats.update(roc_auc=roc_auc, dr_acc=acc)

    return stats.roc_auc, stats.dr_acc


def evaluate_peak_detection(model, val_loader, device, save_folder_name):
    model.eval()

    stats = data_helper.AverageMeter('roc_auc', 'dr_acc')

    with torch.no_grad():
        for test_key, seq, hitframe_gt, direction_gt, cps, n_frames, nfps, picks, user_summary, video_name in val_loader:
            seq_len = len(seq)
            seq_torch = torch.from_numpy(seq).unsqueeze(0).to(device)

            pred_cls, pred_dr = model.predict(seq_torch)
            pred_cls /= pred_cls.max() + 1e-8

            roc_auc = roc_auc_score(hitframe_gt, pred_cls)

            acc = (pred_dr.argmax(1) == direction_gt).sum() / direction_gt.size

            
            stats.update(roc_auc=roc_auc, dr_acc=acc)
        
            # Peak detection
            pred_peaks = find_peaks(pred_cls, height=0.4, distance=15)[0]

            max_peak_count = n_frames / 30

            if len(pred_peaks) > max_peak_count:
                pred_peaks = topk_idx(pred_cls, pred_peaks, round(max_peak_count))
                pred_peaks = np.array(pred_peaks)
            
            peak_count = len(pred_peaks)
            gt_count = hitframe_gt.sum()/5

            # 設置樣式
            sns.set_style("whitegrid")

            # 繪製散點圖
            ax = sns.scatterplot(x=range(len(hitframe_gt)), y=hitframe_gt, color='blue', label='y_true')
            sns.scatterplot(x=pred_peaks, y=pred_cls[pred_peaks], color='red', label='y_peak', ax=ax)

            plt.plot(pred_cls, color='k')
            plt.legend()

            plt.text(5, 1.12, 'GT count: ' + str(gt_count) + ' pred_count: ' + str(peak_count) + ' n_frames: ' + str(n_frames), fontsize=14)
            plt.text(5, 1.07, 'ROC AUC: ' + str(roc_auc), fontsize=14)

            # plt.show()
            video_name_parts = video_name.split('_')
            number = int(video_name_parts[1])
            video_name = '{:05d}'.format((number))
            save_fig_path = os.path.join(save_folder_name, video_name + ".jpg")
            plt.savefig(save_fig_path, format='jpg')
            plt.close()

    return stats.roc_auc, stats.dr_acc

def main():
    args = init_helper.get_arguments()

    # Make saved dir
    save_folder_name = os.path.join(args.save_dir, os.path.basename(args.ckpt_path))
    os.makedirs(save_folder_name, exist_ok=True)

    init_helper.init_logger(args.model_dir, args.log_file)
    init_helper.set_random_seed(args.seed)
    logger.info(vars(args))

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)

    for split_path in args.splits:
        split_path = Path(split_path)
        splits = data_helper.load_yaml(split_path)

        stats = data_helper.AverageMeter('roc_auc', 'dr_acc')

        for split_idx, split in enumerate(splits):
            ckpt_path = data_helper.get_ckpt_path(args.model_dir, split_path, split_idx)
            state_dict = torch.load(str(ckpt_path),
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(state_dict)

            # Load val data
            val_set = data_helper.CustomDataset(split['test_keys'])
            val_loader = data_helper.DataLoader(val_set, shuffle=False)

            roc_auc, dr_acc = evaluate_peak_detection(model, val_loader, args.device, save_folder_name)
            stats.update(roc_auc=roc_auc, dr_acc=dr_acc)

            logger.info(f'{split_path.stem} split {split_idx}: roc_auc: '
                        f'{roc_auc:.4f}, dr_acc: {dr_acc:.4f}')

        logger.info(f'{split_path.stem}: roc_auc: {stats.roc_auc:.4f}, '
                    f'dr_acc: {stats.dr_acc:.4f}')


if __name__ == '__main__':
    main()
