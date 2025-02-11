import os


class Arguments:
    def __init__(self):
        self.use_bgr = False
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
        self.save = "save"
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
        self.train_dir = ""
        self.val_dir = ""
        self.skip_threshold = 1e6
        self.loss = "1*L1"
        self.optimizer = "ADAM"
        self.epochs = 3000
        self.lr = 1e-4
        self.lr_decay = 50
        self.decay_type = "step"
        self.gamma = 0.6
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.epsilon = 1e-8
        self.weight_decay = 0
        self.resume_learning = None

    def set_resume_learning(self, scale, path, epochs):
        self.lr = 1e-8  # 1/2 of initial learning rate; this is set up for finetuning
        self.lr_decay = 100  # double the learning rate decay
        #self.loss = "1*MS-SSIM"  # Now attemoting MS-SSIM based loss
        self.resume_learning = os.path.join(path, "SAN_BI" + str(scale) + "X.pt")
        self.epochs = epochs

    def set_batch(self, batch):
        self.batch_size = batch

    def set_mode_upscale(self, scale, path):
        self.test_only = True
        self.pre_train = os.path.join(path, "SAN_BI" + str(scale) + "X.pt")
        self.scale = scale

    def set_cpu(self):
        self.cpu = True

    def set_use_bgr(self):
        self.use_bgr = True

    def set_mode_train(self, scale, train_dir, val_dir, epochs):
        self.test_only = False
        self.pre_train = "."
        self.scale = scale
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.epochs = epochs

    def push(self):
        self.scale = [self.scale]
        return 0

    def pop(self):
        self.scale = self.scale[0]
