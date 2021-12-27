from mmcv import Config
from mmdet.datasets import build_dataset


def main():
    config = 'configs/fcos/fcos_sar-ship.py'
    cfg = Config.fromfile(config)
    cfg.data.test.img_prefix = cfg.data_root + 'test/'
    dataset = build_dataset(cfg.data.test)
    # in CustomDataset
    dataset.draw_gt_bboxes()


if __name__ == '__main__':
    main()
