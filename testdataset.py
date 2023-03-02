from lib.datasets.roidb import combined_roidb_for_training
from roi_data.loader import RoiDataLoader

roidb, ratio_list, ratio_index = combined_roidb_for_training(
    cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES)
roidb_size = len(roidb)


random.seed(cfg.RNG_SEED)
np.random.seed(cfg.RNG_SEED)
torch.manual_seed(cfg.RNG_SEED)
os.environ['PYTHONHASHSEED'] = str(cfg.RNG_SEED)
if cfg.CUDA:
    torch.cuda.manual_seed(cfg.RNG_SEED)
    torch.cuda.manual_seed_all(cfg.RNG_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

batchSampler = BatchSampler(
    sampler=MinibatchSampler(ratio_list, ratio_index),
    batch_size=args.batch_size,
    drop_last=True
)
dataset = RoiDataLoader(
    roidb,
    cfg.MODEL.NUM_CLASSES,
    training=True)