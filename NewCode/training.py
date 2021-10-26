from arguments import Arguments
from decimal import Decimal
import model
import utility
import loss
import torch
import os
import cv2
import numpy as np

from skimage.metrics import peak_signal_noise_ratio


def convert_regions_to_pytorch(regions):
    for i in range(0, len(regions)):
        regions[i] = np.moveaxis(regions[i], 2, 0)
    return regions


def convert_to_pytorch(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')
    return img


def batch_decomposition(regions_lr, regions_hr, batch_size):
    batches = []
    batch_lr = []
    batch_hr = []
    while len(regions_lr) > 0:
        lr = regions_lr.pop(0)
        hr = regions_hr.pop(0)
        batch_lr.append(lr)
        batch_hr.append(hr)
        if len(batch_lr) >= batch_size:
            batches.append([np.stack(batch_lr, axis=0), np.stack(batch_hr, axis=0)])
            batch_lr = []
            batch_hr = []
    return batches


class Trainer:
    def __init__(self, args: Arguments, train_set, val_set):
        self.args = args
        self.train_set = train_set
        self.val_set = val_set
        self.args.push()
        self.checkpoint = utility.Checkpoint(self.args)
        self.model = model.Model(self.args, self.checkpoint)
        self.loss = loss.Loss(args, self.checkpoint)
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        self.cache = {}
        self.args.pop()

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.checkpoint.dir, 'optimizer.pt'))
            )
            for _ in range(len(self.checkpoint.log)):
                self.scheduler.step()

        self.error_last = 1e8

        print("Precaching training set...")
        self.precache(self.train_set)
        print("Done!")
        print("Precaching validation set...")
        self.precache(self.val_set)
        print("Done!")

    def precache(self, path_list):
        for path in path_list:
            self.load_train_image(path)

    def load_train_image(self, path):
        if path in self.cache:
            return self.cache[path]
        hr = cv2.imread(path)
        lr = cv2.resize(hr, (int(hr.shape[1] / self.args.scale), int(hr.shape[0] / self.args.scale)),
                        interpolation=cv2.INTER_CUBIC)
        region_size = utility.get_max_region_size(lr.shape[0], lr.shape[1])
        lr = convert_to_pytorch(lr)
        regions_lr = convert_regions_to_pytorch(utility.image_decomposition(lr, region_size))
        regions_hr = convert_regions_to_pytorch(utility.image_decomposition(hr, region_size * self.args.scale))
        batches = batch_decomposition(regions_lr, regions_hr, self.args.batch_size)
        self.cache[path] = batches
        return batches

    def prepare(self, l):
        device = torch.device('cpu' if self.args.cpu else 'cuda')

        def _prepare(tensor):
            tensor = torch.from_numpy(tensor)
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(_l) for _l in l]

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_last_lr()[0]

        self.checkpoint.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        batch = 0

        timer_data, timer_model = utility.Timer(), utility.Timer()
        for (i, path) in enumerate(self.train_set):
            for (bi, python_is_a_peace_of_shit) in enumerate(self.load_train_image(path)):
                lr = python_is_a_peace_of_shit[0]
                hr = python_is_a_peace_of_shit[1]
                batch += 1
                timer_data.hold()
                timer_model.tic()
                lr, hr = self.prepare([lr, hr])
                self.optimizer.zero_grad()

                idx_scale = self.args.push()
                sr = self.model(lr, idx_scale)
                self.args.pop()

                loss = self.loss(sr, hr)

                if loss.item() < self.args.skip_threshold * self.error_last:
                    loss.backward()
                    self.optimizer.step()
                else:
                    print('Skip this batch {}! (Loss: {})'.format(i * bi, loss.item()))

                timer_model.hold()

                if batch % 100 == 0:
                    self.checkpoint.write_log('[{}/?]\t{}\t{:.1f}+{:.1f}s'.format(
                        batch,
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()
                    ))

                timer_data.tic()

        self.loss.end_log(batch)
        self.error_last = self.loss.log[-1, -1]
        self.scheduler.step()

    def validate(self):
        epoch = self.scheduler.last_epoch + 1
        self.checkpoint.write_log('\nEvaluation:')
        self.checkpoint.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_val = utility.Timer()
        batch = 0
        with torch.no_grad():
            eval_acc = 0
            for (i, path) in enumerate(self.val_set):
                for (bi, python_is_a_peace_of_shit) in enumerate(self.load_train_image(path)):
                    lr = python_is_a_peace_of_shit[0]
                    hr = python_is_a_peace_of_shit[1]
                    batch += 1
                    lr, hr = self.prepare([lr, hr])

                    idx_scale = self.args.push()
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    self.args.pop()

                    eval_acc += peak_signal_noise_ratio(hr, sr)

                self.checkpoint.log[-1, 0] = eval_acc / batch
                best = self.checkpoint.log.max(0)
                self.checkpoint.write_log(
                    '[{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        path,
                        self.checkpoint.log[-1, 0],
                        best[0][0],
                        best[1][0] + 1
                    )
                )

            self.checkpoint.write_log(
                'Total time: {:.2f}s\n'.format(timer_val.toc()), refresh=True
            )
            self.checkpoint.save(self, epoch, is_best=(best[1][0] + 1 == epoch))


def run(args: Arguments):
    train_set = []
    val_set = []
    files = os.listdir(args.train_dir)
    for f in files:
        train_set.append(os.path.join(args.train_dir, f))
    files = os.listdir(args.val_dir)
    for f in files:
        val_set.append(os.path.join(args.val_dir, f))
    trainer = Trainer(args, train_set, val_set)
    for _ in range(0, args.epochs):
        trainer.train()
        trainer.validate()
