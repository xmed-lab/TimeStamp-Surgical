from io import TextIOBase
import torch
from uncertainty_model import Trainer
from batch_gen import BatchGenerator
import argparse
import random
import time
import os
import numpy as np

from feature_extract.data_util import phase2label_dicts


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_seed(seed):
    # comment out seed to train the model
    if seed is None:  # random
        return 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='casual')
parser.add_argument('--action', default='train', help='two options: train or predict')
parser.add_argument('--seed', type=int, default=20000604)
parser.add_argument('--dataset', help='two dataset: m2cai16, cholec80', choices=['cholec80', 'm2cai16'])
parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epoch')
parser.add_argument('--pseudo', help='how to generate pseudo label')
parser.add_argument('--extract', type=str, help='Feature extraction type')
parser.add_argument('--noisy', action='store_true', help='wheather to generate noisy labels')
parser.add_argument('--uncertainty_warmup_epochs', type=int, default=10)
parser.add_argument('--max_thres', type=float, default=np.log(2), help='uncertainty reject threshould')
parser.add_argument('--smooth', action='store_true', help='whether to use smooth loss')
parser.add_argument('--lambda_smooth', type=float, default=0.15)
parser.add_argument('--entropy', action='store_true', help='whether to use entropy regularization')
parser.add_argument('--lambda_entropy', type=float, default=0.1)
parser.add_argument('--forward_times', type=int, default=10, help='stochastic forward times to MC-dropout')
parser.add_argument('--visualization', action='store_true')
args = parser.parse_args()

num_stages = 2
num_layers = 10
num_f_maps = 64
features_dim = 2048
batch_size = 8
lr = 0.001
num_epochs = args.num_epochs
pseudo = args.pseudo
sample_rate = 1
set_seed(args.seed)

# train_features = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'train_dataset', 'video_feature@2020')
# test_features = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'test_dataset', 'video_feature@2020')
# train_gt_path = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'train_dataset', 'annotation_folder')
# test_gt_path = os.path.join("/home/zxwang/weak-surgical/casual_tcn/", args.dataset, 'test_dataset', 'annotation_folder')
train_features = os.path.join('feature_extract', args.dataset, 'train_dataset', 'video_feature@{}'.format(args.extract))
test_features = os.path.join('feature_extract', args.dataset, 'test_dataset', 'video_feature@{}'.format(args.extract))
train_gt_path = os.path.join('feature_extract', args.dataset, 'train_dataset', 'annotation_folder')
test_gt_path = os.path.join('feature_extract', args.dataset, 'test_dataset', 'annotation_folder')


# Use time data to distinguish output folders in different training
# time_data = '2021-09-26_17-44-18' # turn on this line in evaluation
# time_data = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
model_dir = os.path.join("./models/", args.dataset, 'arch-{}_extract-{}_pseudo-{}'.format(args.arch, args.extract, pseudo))

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("{} dataset {} for single stamp supervision".format(args.action, args.dataset))
print('Batch size is {}, number of stages is {}, sample rate is {}\n'.format(batch_size, num_stages, sample_rate))
print('Extractor: {}, Pseudo labeling: {}'.format(args.extract, args.pseudo))
print('Uncertainty warm up epochs: {}, max threshold: {}'.format(args.uncertainty_warmup_epochs, args.max_thres))
print('Use smooth loss: {}, lambda smooth loss: {}'.format(args.smooth, args.lambda_smooth))
print('Use entropy regularization: {}, lambda entropy: {}'.format(args.entropy, args.lambda_entropy))
phase2label = phase2label_dicts[args.dataset]

num_classes = len(phase2label)
trainer = Trainer(test_features, test_gt_path, phase2label, device, num_stages, num_layers, num_f_maps, features_dim, num_classes, args)

if args.action == "train":
    batch_gen = BatchGenerator(num_classes, phase2label, train_gt_path, train_features, sample_rate)
    # Train the model
    trainer.multi_train(model_dir, batch_gen, batch_size=batch_size, learning_rate=lr)
elif args.action == 'test':
    trainer.predict(model_dir, num_epochs, sample_rate)
else:
    raise NotImplementedError('Invalid action')

print('Extractor: {}, Pseudo labeling: {}'.format(args.extract, args.pseudo))
print('Uncertainty warm up epochs: {}, max threshold: {}'.format(args.uncertainty_warmup_epochs, args.max_thres))
print('Use smooth loss: {}, lambda smooth loss: {}'.format(args.smooth, args.lambda_smooth))
print('Use entropy regularization: {}, lambda entropy: {}'.format(args.entropy, args.lambda_entropy))
