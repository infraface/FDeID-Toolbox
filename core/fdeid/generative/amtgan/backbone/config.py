# from fvcore.common.config import CfgNode
import yaml

class CfgNode(dict):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super(CfgNode, self).__init__()
        self.__dict__ = self
        if init_dict is not None:
            for k, v in init_dict.items():
                if isinstance(v, dict):
                     self[k] = CfgNode(v)
                else:
                    self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def merge_from_file(self, yaml_file):
        with open(yaml_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
        self.merge_from_dict(cfg)

    def merge_from_dict(self, cfg):
        for k, v in cfg.items():
             if k not in self:
                 raise KeyError(f"Key {k} does not exist in config")
             if isinstance(v, dict):
                 if not isinstance(self[k], CfgNode):
                     self[k] = CfgNode(v)
                 else:
                     self[k].merge_from_dict(v)
             else:
                 self[k] = v

    def merge_from_list(self, opts):
        if opts is None: return
        assert len(opts) % 2 == 0
        for i in range(0, len(opts), 2):
            self.merge_from_other_cfg(opts[i], opts[i+1])

    def merge_from_other_cfg(self, key, value):
        keys = key.split('.')
        d = self
        for k in keys[:-1]:
            d = d[k]
        target_type = type(d[keys[-1]])
        if target_type == bool:
             v = value.lower()
             if v == 'true': value = True
             elif v == 'false': value = False
             else: raise ValueError(f"Invalid boolean value: {value}")
        else:
             try:
                 value = target_type(value)
             except ValueError:
                 pass # keep as string if conversion fails
        d[keys[-1]] = value

    def freeze(self):
        pass

    def clone(self):
        import copy
        return copy.deepcopy(self)


_C = CfgNode()


# Paths for logging and saving
_C.LOG = CfgNode()
_C.LOG.LOG_PATH = 'log/'
_C.LOG.SNAPSHOT_PATH = 'snapshot/'
_C.LOG.VIS_PATH = 'visulization/'
_C.LOG.SNAPSHOT_STEP = 1024
_C.LOG.LOG_STEP = 8
_C.LOG.VIS_STEP = 1024

# Data settings
_C.DATA = CfgNode()
_C.DATA.PATH = 'assets/datasets/MT-dataset'
_C.DATA.NUM_WORKERS = 0
_C.DATA.BATCH_SIZE = 1
_C.DATA.IMG_SIZE = 256

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 2e-4
_C.TRAINING.D_LR = 2e-4
_C.TRAINING.H_LR = 2e-4
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = 0.999
_C.TRAINING.C_DIM = 2
_C.TRAINING.G_STEP = 1
_C.TRAINING.NUM_EPOCHS = 50
_C.TRAINING.NUM_EPOCHS_DECAY = 0

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.GAN = 10.0
_C.LOSS.CYCLE = 10.0
_C.LOSS.ADVATTACK = 5.0
_C.LOSS.LAMBDA_HIS_LIP = 1
_C.LOSS.LAMBDA_HIS_SKIN = 0.1
_C.LOSS.LAMBDA_HIS_EYE = 1
_C.LOSS.MAKEUP = 2.0
_C.LOSS.IDT = 5.0

# Model structure
_C.MODEL = CfgNode()
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.D_CONV_DIM = 64
_C.MODEL.G_REPEAT_NUM = 6
_C.MODEL.D_REPEAT_NUM = 3
_C.MODEL.NORM = "SN"
_C.MODEL.WEIGHTS = "checkpoints/"


# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [7, 9]
_C.PREPROCESS.FACE_CLASS = [1, 6]
_C.PREPROCESS.LANDMARK_POINTS = 68

# Postprocessing
_C.POSTPROCESS = CfgNode()
_C.POSTPROCESS.WILL_DENOISE = False

# Device
_C.DEVICE = CfgNode()
_C.DEVICE.device = 'cuda'

# local model zoo
_C.MODELZOO = CfgNode()
_C.MODELZOO.MODELS = ['facenet', 'ir152', 'irse50']
_C.MODELZOO.PATH = 'assets/models'

def get_config():
    return _C.clone()
