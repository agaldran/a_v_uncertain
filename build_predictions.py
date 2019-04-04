import os, argparse
import os.path as osp
import numpy as np
from PIL import Image
from skimage import img_as_float, img_as_ubyte
from skimage.io import imsave
from skimage.transform import resize as sk_resize
from utils.image_preprocessing import correct_illumination, color_transfer
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as tr
import warnings
warnings.filterwarnings("ignore")
from models.unet import UNet


def build_preds(input_im, tg_size):
    img = tr.ToTensor()(resize(input_im)).view(1, 3, *tg_size).to(device)

    with torch.no_grad():
        logits = model(img)
        probas = F.softmax(logits, dim=1)

    width, height = input_im.size
    orig_size = (height, width)
    pred_rgb = np.zeros([*tg_size, 3])
    probas_np = probas[0].detach().cpu().numpy()

    # probas has 4 channels
    pred_vessels = 1 - probas_np[0]
    pred_rgb[:, :, 0] = probas_np[1]
    pred_rgb[:, :, 1] = probas_np[2]
    pred_rgb[:, :, 2] = probas_np[3]
    pretty_pred = (pred_rgb + np.stack(3 * [probas_np[0]], axis=2)).clip(0, 1)
    # recover original resolution
    pred_vessels = sk_resize(pred_vessels, orig_size, anti_aliasing=True, mode='reflect')
    pred_rgb = sk_resize(pred_rgb, orig_size, anti_aliasing=True, mode='reflect')
    pretty_pred = sk_resize(pretty_pred, orig_size, anti_aliasing=True, mode='reflect')

    return pred_vessels, pred_rgb, pretty_pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_ims', metavar='path', required=False, default='retinal_images',
                        help='Images for which we want the A/V predictions to be generated.')
    parser.add_argument('--path_out', metavar='path', required=False, default='results', help='Path for saving results.')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    path_out_ims = osp.join(args.path_out, 'uncertainty/')
    path_out_vessels = osp.join(args.path_out, 'uncertainty_vessels/')
    path_out_pretty_preds = osp.join(args.path_out, 'pretty_preds/')
    path_test_ims = args.path_ims

    os.makedirs(path_out_ims, exist_ok=True)
    os.makedirs(path_out_vessels, exist_ok=True)
    os.makedirs(path_out_pretty_preds, exist_ok=True)
    os.makedirs(path_test_ims, exist_ok=True)

    path_to_check_point = 'models/checkpoints_uncertainty/model_final.pth.tar'
    ref_im = Image.open('models/36_training.tif')


    print('Predicting with TTA...')

    flip_vert = tr.RandomVerticalFlip(p=1)
    flip_horz = tr.RandomHorizontalFlip(p=1)
    num_classes = 4
    model = UNet(num_classes)
    checkpoint = torch.load(path_to_check_point)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})".format(path_to_check_point, checkpoint['epoch']))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tg_size = (512, 512)
    resize = tr.Resize(tg_size)
    transforms = tr.Compose([resize, tr.ToTensor()])

    im_list = os.listdir(path_test_ims)

    for i in tqdm(range(len(im_list))):
        im_name = im_list[i]

        # predictions for input image
        input_im = Image.open(osp.join(path_test_ims, im_name))
        # illumination correction
        illum_restored = img_as_ubyte(correct_illumination(img_as_float(input_im)))
        # color transfer
        input_im = Image.fromarray(color_transfer(ref_im, illum_restored))
        # compute predictions
        pred_vessels, pred_rgb, pretty_pred = build_preds(input_im, tg_size)

        # predictions for flipped_lr image
        tta = lambda a: np.fliplr(a)
        input_im_lr = Image.fromarray(tta(np.array(input_im)))
        pred_vessels_lr, pred_rgb_lr, pretty_pred_lr = build_preds(input_im_lr, tg_size)
        pred_vessels_lr, pred_rgb_lr, pretty_pred_lr = tta(pred_vessels_lr), tta(pred_rgb_lr), tta(pretty_pred_lr)

        # predictions for flipped_ud image
        tta = lambda a: np.flipud(a)
        input_im_ud = Image.fromarray(tta(np.array(input_im)))
        pred_vessels_ud, pred_rgb_ud, pretty_pred_ud = build_preds(input_im_ud, tg_size)
        pred_vessels_ud, pred_rgb_ud, pretty_pred_ud = tta(pred_vessels_ud), tta(pred_rgb_ud), tta(pretty_pred_ud)

        # predictions for flipped_lr_ud image
        tta = lambda a: np.flipud(np.fliplr((a)))
        input_im_lr_ud = Image.fromarray(tta(np.array(input_im)))
        pred_vessels_lr_ud, pred_rgb_lr_ud, pretty_pred_lr_ud = build_preds(input_im_lr_ud, tg_size)
        pred_vessels_lr_ud, pred_rgb_lr_ud, pretty_pred_lr_ud = \
            tta(pred_vessels_lr_ud), tta(pred_rgb_lr_ud), tta(pretty_pred_lr_ud)

        # average predictions
        pred_rgb = 0.25 * (pred_rgb + pred_rgb_lr + pred_rgb_ud + pred_rgb_lr_ud)
        pred_vessels = 0.25 * (pred_vessels + pred_vessels_lr + pred_vessels_ud + pred_vessels_lr_ud)
        pretty_pred = 0.25 * (pretty_pred + pretty_pred_lr + pretty_pred_ud + pretty_pred_lr_ud)

        # save to disk
        pred_rgb = (255 * pred_rgb).astype(np.uint8)
        imsave(path_out_ims + im_name[:-4] + '.png', pred_rgb)
        pred_vessels = (255 * pred_vessels).astype(np.uint8)
        imsave(path_out_vessels + im_name[:-4] + '.png', pred_vessels)
        pretty_pred = (255 * pretty_pred).astype(np.uint8)
        imsave(path_out_pretty_preds + im_name[:-4] + '_pretty.png', pretty_pred)

    print('Done')
