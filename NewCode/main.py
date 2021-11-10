import argparse

import upscale
import training

from arguments import Arguments


def main():
    parser = argparse.ArgumentParser(description='SAN network')
    parser.add_argument('--upscale', help="Upscale single image", action='store_true')
    parser.add_argument('--cpu', help="Use CPU", action='store_true')
    parser.add_argument('--use-bgr', help="Consider image to already be in BGR format when testing", action='store_true')
    parser.add_argument('--train', help="Start training", action='store_true')
    parser.add_argument('--train-dir', type=str, help="image path")
    parser.add_argument('--val-dir', type=str, help="image path")
    parser.add_argument('--scale', default='2', choices=['2', '3', '4', '8'], help='super resolution scale')
    parser.add_argument('--image', type=str, help="image path")
    parser.add_argument('--batch', type=int, default=8, help="image batch size")
    parser.add_argument('--model-path', type=str, help="path to the model .pt files")
    cmd = parser.parse_args()
    args = Arguments()
    if cmd.use_bgr:
        args.set_use_bgr()
    if cmd.upscale:
        args.set_mode_upscale(int(cmd.scale), cmd.model_path)
    elif cmd.train:
        if cmd.model_path:
            args.set_resume_learning(int(cmd.scale), cmd.model_path)
        args.set_mode_train(int(cmd.scale), cmd.train_dir, cmd.val_dir)
    if cmd.cpu:
        args.set_cpu()
    args.set_batch(cmd.batch)
    if cmd.upscale:
        upscale.run(args, cmd.image)
    elif cmd.train:
        training.run(args)
    print("I ran successfully!")


main()
