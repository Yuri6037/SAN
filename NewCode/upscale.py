import argparse
import os
import model
import utility
import cv2
import torch
import numpy as np

from arguments import Arguments

def prepare(args, l):
    device = torch.device('cpu' if args.cpu else 'cuda')

    def _prepare(tensor):
        tensor = torch.from_numpy(tensor)
        if args.precision == 'half':
            tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]


def convert_region_to_pytorch(img):
    img = np.moveaxis(img, 2, 0)
    return img


def convert_to_pytorch(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    return img


def inline_reconstruct_array(regions):
    lst = []
    for i in range(regions.shape[0]):
        region = regions[i, ...]
        region = np.moveaxis(region, 0, 2)
        lst.append(region)
    return lst


def run(args: Arguments, path: str):
    img = cv2.imread(path)
    name = os.path.basename(path)
    region_size = utility.get_max_region_size(img.shape[0], img.shape[1])

    # Upscale with Bicubic
    bicubic_image = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("./result/bicubic/" + name, bicubic_image)

    # Upscale with SAN
    print("Loading model...")
    idx_scale = args.push()
    ckpt = utility.Checkpoint(args)
    mdl = model.Model(args, ckpt)
    print("Upscaling single image...")
    LR_Image = convert_to_pytorch(img)
    regions = utility.image_decomposition(LR_Image, region_size)
    for i in range(0, len(regions)):
        regions[i] = convert_region_to_pytorch(regions[i])
    regions = np.stack(regions, axis=0)
    print("LR Decomposed: %s" % str(regions.shape))
    lst = []
    for LR_Batch in np.array_split(regions, int(regions.shape[0] / args.batch_size) + 1, axis=0):
        print(LR_Batch.shape)
        LR_Batch = prepare(args, [LR_Batch])[0]
        HR_Batch = mdl(LR_Batch, idx_scale)
        HR_Batch = utility.quantize(HR_Batch, args.rgb_range)
        lst.append(HR_Batch.detach().cpu().numpy())
    args.pop()
    regions = np.concatenate(lst, axis=0)
    print("HR Decomposed: %s" % str(regions.shape))
    regions = inline_reconstruct_array(regions)
    HR_Image = np.zeros((img.shape[0] * args.scale, img.shape[1] * args.scale, 3), np.uint8)
    HR_Image = utility.image_recomposition(HR_Image, region_size * args.scale, regions)
    print("HR: %s" % str(HR_Image.shape))
    HR_Image = HR_Image.astype(np.uint8)
    HR_Image = cv2.cvtColor(HR_Image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./result/san/" + name, HR_Image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
