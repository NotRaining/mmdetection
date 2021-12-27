import cv2
from glob import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
import xml.etree.ElementTree as ET
import math

# color scheme: https://www.jianshu.com/p/a2bfbee2fbec
color_scheme = dict(red='#c44e52',
                    orange='#dd8452',
                    yellow='#ccb974',
                    green='#55a868',
                    blue='#6195C8',
                    indigo='#4c72b0',
                    purple='#8172b3')
cs = dict(r='#c44e52',
          o='#dd8452',
          y='#ccb974',
          g='#55a868',
          b='#6195C8',
          i='#4c72b0',
          p='#8172b3')


def calc_mean_std(imgs: list, type='images'):
    means, stds = [0, 0, 0], [0, 0, 0]
    for img in imgs:
        # (h, w, c)
        assert len(img.shape) == 3
        for j in range(3):
            means[j] += np.mean(img[:, :, j])
            stds[j] += np.std(img[:, :, j])

    mean = [round(x / len(imgs), 2) for x in means]
    std = [round(x / len(imgs), 2) for x in stds]
    print(f'mean of {type}: {mean}, std of {type}: {std}')


def analyze_images(img_dir):
    img_paths = glob(os.path.join(img_dir, '*'))
    num_imgs = len(img_paths)
    print(f'number of images: {num_imgs}')
    imgs = [cv2.imread(path) for path in img_paths]
    calc_mean_std(imgs, type='images')
    # SSDD: mean=39.75, std=27.72
    # SAR-Ship-Dataset: mean=21.46, std=24.40


def analyze_objects(img_dir, classes, label_type='xml'):
    assert label_type == 'xml'
    img_paths = glob(os.path.join(img_dir, '*'))
    ext_name = os.path.basename(img_paths[0]).split('.')[1]
    label_paths = [img_path.replace('JPEGImages', 'Annotations').replace(ext_name, 'xml') for img_path in img_paths]
    hw_list, cat_list, patch_list = [], [], []

    for label_path, img_path in zip(label_paths, img_paths):
        tree = ET.parse(label_path)
        img = cv2.imread(img_path)

        for obj in tree.findall('object'):
            category = obj.find('name').text
            cat_list.append(category)

            bndbox = obj.find('bndbox')
            # int-str or float-str to number
            xmin = int(eval(bndbox.find('xmin').text))
            xmax = int(eval(bndbox.find('xmax').text))
            ymin = int(eval(bndbox.find('ymin').text))
            ymax = int(eval(bndbox.find('ymax').text))
            w = xmax - xmin
            h = ymax - ymin
            hw_list.append((h, w))
            patch_list.append(img[ymin:ymax, xmin:xmax, :])

    num_objs = len(hw_list)
    print(f'number of objects: {num_objs}')
    calc_mean_std(patch_list, type='objects')

    # mpl.rcParams['font.size']=10
    # use style
    # print(plt.style.available)
    # plt.style.use('seaborn')
    # plt.style.use('ggplot')
    fig, ax = plt.subplots(2, 2, figsize=[12, 9])
    ax = ax.reshape(-1)  # 2*2

    # analyze aspect ratio
    aspect_ratios = [float(hw[1]) / float(hw[0]) for hw in hw_list]
    right = math.ceil(max(aspect_ratios))
    num_bins = 20
    n, bins_limits, patches = ax[0].hist(aspect_ratios,
                                         bins=np.linspace(0, right, num_bins + 1),
                                         histtype='bar',
                                         color=cs['i'],
                                         edgecolor='w', )
    offset = right / (num_bins * 2)
    bins_limits = bins_limits[:num_bins] + offset
    n = np.concatenate(([0], n))
    bins_limits = np.concatenate(([0], bins_limits))
    ax[0].plot(bins_limits, n, linestyle='-', marker='o', markersize=5, color=cs['o'])
    ax[0].grid(linestyle='--', linewidth=0.5)
    # ax[0].minorticks_on()
    # ax[0].xaxis.set_major_locator(MultipleLocator(1))
    ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
    # ax[0].xaxis.set_minor_formatter(FuncFormatter(lambda x, pos: '%.2f' % x if x % 1 else ''))
    ax[0].set_xlim([0, right])
    ax[0].set_xlabel('aspect ratio', fontsize=12, fontweight='normal', ha='center')
    ax[0].set_ylabel('instances', fontsize=12)

    # analyze area
    area_type = ['small', 'medium', 'large']
    area_list = [h * w for h, w in hw_list]
    areas = np.array(area_list)
    num_small_area = (areas <= 32 ** 2).sum()
    num_medium_area = ((areas > 32 ** 2) & (areas <= 96 ** 2)).sum()
    num_large_area = (areas > 96 ** 2).sum()
    pct_area = np.array([num_small_area, num_medium_area, num_large_area]) / num_objs * 100
    ax[1].bar(x=range(1, 4), height=pct_area, width=0.5, color=cs['b'])
    ax[1].set_xticks(range(1, 4))
    ax[1].set_xticklabels(area_type)  # fontproperties={'family': 'Times New Roman'}
    ax[1].set_xlabel('area', fontsize=12)
    ax[1].set_yticks(np.linspace(0, 100, 11, endpoint=True))
    ax[1].set_ylabel('percentage(%)', fontsize=12)
    for x, y in enumerate(pct_area):
        s = '%.2f' % y
        ax[1].text(x + 1, y + 1, f'{s}%', ha='center')
    # ax[1].set(xticks=range(1, 4),
    #           xticklabels=['small', 'medium', 'big'],
    #           xlabel='area',
    #           ylabel='number')

    # analyze class
    cat2id = {name: i for i, name in enumerate(classes)}
    ids = [cat2id[cat] for cat in cat_list]
    pct_cls = np.bincount(ids) / num_objs * 100
    ax[2].bar(x=range(1, len(cat2id) + 1), height=pct_cls, width=0.5, color=cs['b'])
    ax[2].set_xticks(range(1, len(cat2id) + 1))
    ax[2].set_xticklabels(classes, rotation=45)
    ax[2].set_xlabel('class', fontsize=12)
    ax[2].set_ylabel('percentage(%)', fontsize=12)
    for x, y in enumerate(pct_cls):
        s = '%.2f' % y
        ax[2].text(x + 1, y + 1, f'{s}%', ha='center')

    # save and show
    # fig.savefig('')
    fig.show()


if __name__ == '__main__':
    img_dir = '/home/not-raining/workspace/datasets/SSDD/JPEGImages/'
    classes = ['ship']
    analyze_images(img_dir)
    analyze_objects(img_dir, classes, label_type='xml')
