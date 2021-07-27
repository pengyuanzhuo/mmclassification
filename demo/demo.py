from argparse import ArgumentParser
import os
import os.path as osp
import glob

import cv2
from mmcls.apis import inference_model, init_model, show_result_pyplot
import mmcv
import matplotlib.pyplot as plt


def get_images(data_dir):
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(data_dir, '*.{}'.format(ext))))
    return files


def get_images_from_list(datalist, img_prefix=None):
    files = []
    with open(datalist, 'r') as f:
        for line in f:
            image_name, _ = line.strip().split(',')
            if image_name == 'image_name':
                continue
            img_path = image_name if img_prefix is None else osp.join(img_prefix, image_name)
            files.append(img_path)
    return files


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--prefix', help='imgprefix')
    parser.add_argument('--outdir', help='draw dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    if osp.isdir(args.img):
        images = get_images(args.img)
    else:
        images = get_images_from_list(args.img, args.prefix)
    with open('result.csv', 'w') as f:
        f.write('image_name,label\n')
        for img in images:
            # test a single image
            result = inference_model(model, img)
            img_name = osp.basename(img)
            label = result['pred_class']
            f.write('{},{}\n'.format(img_name, label))

            # show the results
            if hasattr(model, 'module'):
                model = model.module
            img = model.show_result(img, result, show=False)
            dst_path = osp.join(args.outdir, img_name)
            cv2.imwrite(dst_path, img)


if __name__ == '__main__':
    main()

