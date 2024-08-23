import argparse
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import datasets.MVTec
import datasets.BTAD
import datasets.WFDD
import datasets.WFT
import datasets.DTD_sys
from einops import rearrange
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./result")
    parser.add_argument('--dataset', choices=['MVTec', 'BTAD', 'WFDD', 'WFT', 'DTD-sys'], default='MVTec')
    return parser.parse_args()

def main():
    args = parse_args()
    print('pwd=', os.getcwd())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset == 'MVTec':
        dataset = datasets.MVTec
    elif args.dataset == 'BTAD':
        dataset = datasets.BTAD
    elif args.dataset == 'WFDD':
        dataset = datasets.WFDD
    elif args.dataset == 'WFT':
        dataset = datasets.WFT
    elif args.dataset == 'DTD-sys':
        dataset = datasets.DTD_sys

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer2[3].register_forward_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in dataset.CLASS_NAMES:
        start_time = time.time()
        test_dataset = dataset.Dataset(class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

        gt_list = []
        gt_mask_list = []
        test_imgs = []
        score_map_list = []
        scores = []
        cut_surrounding = 32

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask[:, :, cut_surrounding:x.shape[2] - cut_surrounding,
                                cut_surrounding:x.shape[2] - cut_surrounding].cpu().detach().numpy().astype(int))
            features = get_feature(model, x, device, outputs)
            m = torch.nn.AvgPool2d(3, 1, 1)
            gallery2 = rearrange(m(features[0]), 'i c h w ->i  (h w) c').unsqueeze(1).to('cpu').detach().numpy().copy()
            heatMap2 = calc_score(gallery2, gallery2, 0)

            for imgID in range(x.shape[0]):
                cut2 = 3
                newHeat = interpolate_scoremap(imgID, heatMap2, cut2, x.shape[2])
                newHeat = gaussian_filter(newHeat.squeeze().cpu().detach().numpy(), sigma=4)
                newHeat = torch.from_numpy(newHeat.astype(np.float32)).clone().unsqueeze(0).unsqueeze(0)

                score_map_list.append(newHeat[:, :, cut_surrounding:x.shape[2]-cut_surrounding,
                                      cut_surrounding:x.shape[2] - cut_surrounding])
                scores.append(score_map_list[-1].max().item())

        ##################################################
        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        # calculate per-pixel level ROCAUC
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        exp_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(exp_path, exist_ok=True)

        img_log_txt = open(os.path.join(exp_path, 'img_auroc.txt'), 'a')
        img_log_txt.write(f"{roc_auc}\n")
        img_log_txt.close()
        class_txt = open(os.path.join(exp_path, 'class_name.txt'), 'a')
        class_txt.write(f"{class_name}\n")
        class_txt.close()
        pix_log_txt = open(os.path.join(exp_path, 'pxl_auroc.txt'), 'a')
        pix_log_txt.write(f"{per_pixel_rocauc}\n")
        pix_log_txt.close()

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, exp_path, class_name, cut_surrounding)

        fig.tight_layout()
        fig.savefig(os.path.join(exp_path, 'roc_curve.png'), dpi=100)

        elapsed_time = time.time() - start_time
        time_log_txt = open(os.path.join(exp_path, 'time_log.txt'), 'a')
        time_log_txt.write(f"{elapsed_time}\n")
        time_log_txt.close()


def interpolate_scoremap(imgID, heatMap, cut,imgshape):
    blank = torch.ones_like(heatMap[imgID, :, :]) * heatMap[imgID, :, :].min()
    blank[cut:heatMap.shape[1] - cut, cut:heatMap.shape[1] - cut] = heatMap[imgID, cut:heatMap.shape[1] - cut,
                                                                    cut:heatMap.shape[1] - cut]
    return F.interpolate(blank[:, :].unsqueeze(0).unsqueeze(0), size=imgshape, mode='bilinear', align_corners=False)


def get_feature(model, img, device, outputs):
    with torch.no_grad():
        _ = model(img.to(device))

    layer2_feature = outputs[-1]

    outputs.clear()
    return [layer2_feature]


def calc_score(test, gallery, layerID):

    heatmap = np.zeros((test.shape[0], test.shape[2]))
    for imgID in range(test.shape[0]):
        nbrs = NearestNeighbors(n_neighbors=400, algorithm='ball_tree').fit(gallery[imgID, layerID, :, :])
        distances, _ = nbrs.kneighbors(test[imgID, layerID, :, :])
        heatmap[imgID, :] = np.mean(distances, axis=1)
        heatmap = torch.from_numpy(heatmap.astype(np.float32)).clone()
    dim = int(np.sqrt(test.shape[2]))
    return heatmap.reshape(test.shape[0], dim, -1)


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         exp_path, class_name,cut_pixel):
    for t_idx in range(len(gt_mask_list)):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        heat = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy().copy()
        test_pred = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy()
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img =test_pred_img[cut_pixel:test_img.shape[0]-cut_pixel,cut_pixel:test_img.shape[0]-cut_pixel, :]
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat)
        ax_img[2].title.set_text('HeatMap')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(exp_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(exp_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()

