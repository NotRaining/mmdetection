import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math

color_scheme = dict(red='#c44e52',
                    orange='#dd8452',
                    yellow='#ccb974',
                    green='#55a868',
                    blue='#64b5cd',
                    indigo='#4c72b0',
                    purple='#8172b3')
cs = dict(r='#c44e52',
          o='#dd8452',
          y='#ccb974',
          g='#55a868',
          b='#64b5cd',
          i='#4c72b0',
          p='#8172b3')


def load_json_logs(json_log):
    train_info, val_info = [], []
    with open(json_log, 'r') as log_file:
        for line in log_file:
            log = json.loads(line.strip())
            if 'train' in log.values():
                train_info.append(dict(epoch=log['epoch'], iter=log['iter'], lr=log['lr'],
                                       loss=log['loss'], loss_cls=log['loss_cls'], loss_bbox=log['loss_bbox']))
            if 'val' in log.values():
                val_info.append(dict(epoch=log['epoch'], iter=log['iter'], AP50=log['bbox_mAP_50'],
                                     mAP=log['bbox_mAP']))

    return train_info, val_info


def plot_curves(train_info, val_info):
    iters_per_epoch = val_info[0]['iter']
    losses = [info['loss'] for info in train_info]
    cls_losses = [info['loss_cls'] for info in train_info]
    bbox_losses = [info['loss_bbox'] for info in train_info]
    lr = [info['lr'] for info in train_info]
    iters = [(info['epoch'] - 1) * iters_per_epoch + info['iter'] for info in train_info]
    AP50 = [info['AP50'] for info in val_info]
    mAP = [info['mAP'] for info in val_info]
    iters_val = [info['epoch'] * info['iter'] for info in val_info]

    # matplotlibrc
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    fig, ax = plt.subplots(2, 2, figsize=[12, 9])
    ax = ax.reshape(-1)

    ax[0].plot(iters, losses, color=cs['r'], label='total_loss')
    # ax[0].plot(iters, cls_losses, color=cs['i'], label='cls_loss')
    # ax[0].plot(iters, bbox_losses, color=cs['g'], label='bbox_loss')
    max_iter = max(iters)
    power = len(str(max_iter)) - 1
    right = math.ceil(max_iter / 10 ** power) * 10 ** power
    ax[0].set_xlim(0, right)
    ax[0].set_xlabel('iterations', fontsize=12)
    ax[0].set_ylim(0)
    ax[0].set_ylabel('loss', fontsize=12)
    # ax[0].grid(ls='--', lw=0.5)
    ax[0].legend()

    ax[1].plot(iters_val, AP50, color=cs['i'], label='AP50')
    max_iter = max(iters_val)
    power = len(str(max_iter)) - 1
    right = math.ceil(max_iter / 10 ** power) * 10 ** power
    ax[1].set_xlim(0, right)
    labels = [int(x / 10 ** power) for x in ax[1].get_xticks()]
    ax[1].set_xticklabels(labels)
    ax[1].set_xlabel(f'iterations(10^{power})', fontsize=12)
    ax[1].set_ylim(0)
    ax[1].set_yticks(np.linspace(0, 1, 6))
    ax[1].set_ylabel('AP', fontsize=12)
    ax[1].legend(loc=4)

    for i, (k, v) in enumerate(color_scheme.items()):
        ax[2].plot(range(0, 10), i * np.arange(0, 10), c=v, label=k)
    ax[2].legend()

    fig.savefig('training.png')
    fig.show()


if __name__ == '__main__':
    json_log = '../../work_dirs/fcos_r50_ssdd_8-1-0.02/20211006_201009.log.json'
    plot_curves(*load_json_logs(json_log=json_log))
