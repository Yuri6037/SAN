import os
import cv2

from TrainCode.data import srdata

class Custom(srdata.SRData):
    def __init__(self, args, train=True):
        args.ext = "raw"
        super(Custom, self).__init__(args, train)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        if self.train:
            files = os.listdir(self.train_dir)
        else:
            files = os.listdir(self.test_dir)
        for f in files:
            hr = cv2.imread(f)
            for (i, scale) in enumerate(self.scale):
                lr = cv2.resize(hr, (int(hr.shape[1] / scale), int(hr.shape[0] / scale)),
                                interpolation=cv2.INTER_CUBIC)
                list_lr[i].append(lr)
            list_hr.append(hr)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        self.train_dir = os.path.join(self.apath, self.args.data_train)
        self.test_dir = os.path.join(self.apath, self.args.data_test)
