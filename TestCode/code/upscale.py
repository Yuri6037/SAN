import argparse
import os
import model
import utility
import cv2
import torch

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
        self.print_model = True

def prepare(args, l):
    device = torch.device('cpu' if args.cpu else 'cuda')
    def _prepare(tensor):
        tensor = torch.from_numpy(tensor)
        if args.precision == 'half':
            tensor = tensor.half()
        return tensor.to(device)
    return [_prepare(_l) for _l in l]

def run(args, path):
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
    LR_Image = prepare(args, [img])[0]
    HR_Image = mdl(LR_Image, 0)
    HR_Image = utility.quantize(HR_Image, args.rgb_range)
    cv2.imwrite("./result/san/" + name, HR_Image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Upscale single image')
    parser.add_argument('--scale', default='2', choices=['2', '3', '4', '8'], help='super resolution scale')
    parser.add_argument('--image', type=str, help="image path")
    parser.add_argument('--model-path', type=str, help="path to the model .pt files")
    cmd = parser.parse_args()
    args = PregenArgs()
    args.pre_train = os.path.join(cmd.model_path, "SAN_BI" + cmd.scale + "X.pt")
    args.scale = int(cmd.scale)
    run(args, cmd.image)
    print("I ran successfully!")

main()
