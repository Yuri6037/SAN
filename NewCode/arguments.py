import os

class Arguments:
    def __init__(self):
        self.model = "san"
        self.n_resgroups = 20
        self.n_resblocks = 10
        self.n_feats = 64
        self.reset = True
        self.chop = True
        self.test_only = False
        self.precision = "single"
        self.cpu = False  # Always run on CUDA by default
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
        self.batch_size = 8

    def set_batch(self, batch):
        self.batch_size = batch

    def set_mode_upscale(self, scale, path):
        self.test_only = True
        self.pre_train = os.path.join(path, "SAN_BI" + str(scale) + "X.pt")
        self.scale = scale

    def set_cpu(self):
        self.cpu = True

    def set_mode_train(self, scale):
        self.test_only = False
        self.pre_train = None
        self.scale = scale

    def push(self):
        self.scale = [self.scale]
        return 0

    def pop(self):
        self.scale = self.scale[0]
