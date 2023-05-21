import logging

import torch
from sklearn.metrics import roc_auc_score

from anchor_free.dsnet_af import DSNetAF
from anchor_free.losses import calc_dr_loss, calc_cls_loss
from evaluate import evaluate
from helpers import data_helper

logger = logging.getLogger()


def train(args, split, save_path):
    model = DSNetAF(base_model=args.base_model, num_feature=args.num_feature,
                    num_hidden=args.num_hidden, num_head=args.num_head)
    model = model.to(args.device)

    model.train()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr,
                                 weight_decay=args.weight_decay)

    max_roc_auc = -1
    max_dr_acc = -1

    train_set = data_helper.CustomDataset(split['train_keys'])
    train_loader = data_helper.DataLoader(train_set, shuffle=True)

    val_set = data_helper.CustomDataset(split['test_keys'])
    val_loader = data_helper.DataLoader(val_set, shuffle=False)

    for epoch in range(args.max_epoch):
        model.train()
        stats = data_helper.AverageMeter('loss', 'cls_loss', 'dr_loss', 'roc_auc', 'dr_acc')
        idx = 0
        for i, (_, seq, hitframe_gt, direction_gt, _, _, _, _, _, _) in enumerate(train_loader):

            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(args.device)
            if torch.isnan(x).any():
                print(x.shape)

            pred_cls, pred_dr = model(x)
            if torch.isnan(pred_cls).any():
                print(pred_cls.shape)
            cls_label = torch.tensor(hitframe_gt, dtype=torch.float32).to(args.device)
            dr_label = torch.tensor(direction_gt, dtype=torch.float32).to(args.device)
            
            cls_loss = calc_cls_loss(pred_cls, cls_label, args.cls_loss)
            dr_loss = calc_dr_loss(pred_dr, dr_label)

            loss = cls_loss + dr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_cls_detach = pred_cls.detach().cpu().numpy()
            roc_auc = roc_auc_score(hitframe_gt, pred_cls_detach)

            pred_dr_detach = pred_dr.detach().cpu().numpy()
            acc = (pred_dr_detach.argmax(1) == direction_gt).sum() / direction_gt.size

            stats.update(loss=loss.item(), cls_loss=cls_loss.item(),
                         dr_loss=dr_loss.item(), roc_auc=roc_auc.item(), dr_acc=acc.item())
            idx += 1
            
        val_roc_auc, val_dr_acc = evaluate(model,  val_loader, args.device)

        if epoch%10 == 0:
            if max_roc_auc < val_roc_auc:
                max_roc_auc = val_roc_auc
                torch.save(model.state_dict(), str(save_path))
            elif max_dr_acc < val_dr_acc:
                max_dr_acc = val_dr_acc
                torch.save(model.state_dict(), str(save_path))

        logger.info(f'Epoch: {epoch}/{args.max_epoch} '
                    f'Loss: {stats.cls_loss:.4f}/{stats.dr_loss:.4f}/{stats.loss:.4f} '
                    f'V_set ROCAUC: {val_roc_auc:.4f} ' 
                    f'T_set ROCAUC: {stats.roc_auc:.4f} '
                    f'V_set DR_ACC: {val_dr_acc:.4f} ' 
                    f'T_set DR_ACC: {stats.dr_acc:.4f}')

    return max_roc_auc
