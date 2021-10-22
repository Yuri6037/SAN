import argparse
import os
import model
import utility
import cv2
import torch
import numpy as np

REGION_SIZE = 16

class PregenArgs:
    def __init__(self):
        self.model = "san"
        self.n_resgroups = 20
        self.n_resblocks = 10
        self.n_feats = 64
        self.reset = True
        self.chop = True
        self.test_only = True
        self.precision = "single"
        self.cpu = False  # Always run on CUDA
        self.pre_train = ""
        self.scale = 2
        self.load = "."
        self.save = "."
        self.degradation = "BI"
        self.testset = "fuckyou"
        self.rgb_range = 255
        self.self_ensemble = False
        self.n_GPUs = 1
        self.save_models = False
        self.resume = 0
        self.reduction = 16
        self.n_colors = 3
        self.res_scale = 1
        self.print_model = False

def prepare(args, l):
    device = torch.device('cpu' if args.cpu else 'cuda')
    def _prepare(tensor):
        tensor = torch.from_numpy(tensor)
        if args.precision == 'half':
            tensor = tensor.half()
        return tensor.to(device)
    return [_prepare(_l) for _l in l]

#def convert_to_pytorch(args, img):
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    img = img.astype('float32')
#    img = np.moveaxis(img, 2, 0)
#    LR_Image = prepare(args, [img])[0]
#    return LR_Image

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
        region = regions[i,...]
        region = np.moveaxis(region, 0, 2)
        lst.append(region)
    return lst

#def reconstruct_image(tensor):
#    tensor = tensor.squeeze(0)
#    ndarr = tensor.detach().cpu().numpy()
#    ndarr = np.moveaxis(ndarr, 0, 2)
#    ndarr = ndarr.astype(np.uint8)
#    ndarr = cv2.cvtColor(ndarr, cv2.COLOR_RGB2BGR)
#    return ndarr

def run(args, path, batch):
    img = cv2.imread(path)
    name = os.path.basename(path)

    # Upscale with Bicubic
    bicubic_image = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("./result/bicubic/" + name, bicubic_image)

    # Upscale with SAN
    print("Loading model...")
    args.scale = [args.scale]
    ckpt = utility.checkpoint(args)
    mdl = model.Model(args, ckpt)
    print("Upscaling single image...")
    LR_Image = convert_to_pytorch(img)
    regions = utility.image_decomposition(LR_Image, REGION_SIZE)
    for i in range(0, len(regions)):
        regions[i] = convert_region_to_pytorch(regions[i])
    regions = np.stack(regions, axis=0)
    print("LR Decomposed: %s" % str(regions.shape))
    lst = []
    for LR_Batch in np.array_split(regions, int(regions.shape[0] / batch) + 1, axis=0):
        print(LR_Batch.shape)
        LR_Batch = prepare(args, [LR_Batch])[0]
        HR_Batch = mdl(LR_Batch, 0)
        HR_Batch = utility.quantize(HR_Batch, args.rgb_range)
        lst.append(HR_Batch.detach().cpu().numpy())
    regions = np.concatenate(lst, axis=0)
    print("HR Decomposed: %s" % str(regions.shape))
    regions = inline_reconstruct_array(regions)
    HR_Image = np.zeros((img.shape[0] * args.scale[0], img.shape[1] * args.scale[0], 3), np.uint8)
    HR_Image = utility.image_recomposition(HR_Image, REGION_SIZE * args.scale[0], regions)
    print("HR: %s" % str(HR_Image.shape))
    HR_Image = HR_Image.astype(np.uint8)
    HR_Image = cv2.cvtColor(HR_Image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./result/san/" + name, HR_Image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Upscale single image')
    parser.add_argument('--scale', default='2', choices=['2', '3', '4', '8'], help='super resolution scale')
    parser.add_argument('--image', type=str, help="image path")
    parser.add_argument('--batch', type=int, default=8, help="image batch size")
    parser.add_argument('--model-path', type=str, help="path to the model .pt files")
    cmd = parser.parse_args()
    args = PregenArgs()
    args.pre_train = os.path.join(cmd.model_path, "SAN_BI" + cmd.scale + "X.pt")
    args.scale = int(cmd.scale)
    run(args, cmd.image, cmd.batch)
    print("I ran successfully!")

main()
