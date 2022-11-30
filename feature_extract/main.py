import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from easydict import EasyDict
import os
import argparse
import numpy as np
import random
from tqdm import tqdm

from data_util import RandomCrop, phase2label_dicts, stats_dict, PureTimestampDataset, FullDataset
from model import inception_v3, SemiNetwork
from models import get_ramp_up, get_augmenter, get_mixmatch_function, interleave
from loss import (build_supervised_loss, build_unsupervised_loss, build_pair_loss)
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 20000604
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', choices=['train_only', 'train_cps', 'train_simple', 'train_mixmatch', 'extract_only', 'extract_cps', 'extract_simple'])
parser.add_argument('--dataset', default="cholec80", choices=['cholec80','m2cai16'])
parser.add_argument('--target', type=str, default='train_set')
parser.add_argument('--epoch', type=int, default=10)
args = parser.parse_args()

epochs = args.epoch
val_interval = epochs

def split_classfier_params(model, classifier_prefix):
    if not isinstance(classifier_prefix, Set):
        classifier_prefix = {classifier_prefix}
    classifier_prefix = tuple(sorted(f"{prefix}." for prefix in classifier_prefix))
    embedder_weights = []
    classifier_weights = []
    for k, v in model.named_parameters():
        if k.startswith(classifier_prefix):
            classifier_weights.append(v)
        else:
            embedder_weights.append(v)
    return embedder_weights, classifier_weights


def get_trainable_params(model, learning_rate, feature_learning_rate=None, classifier_prefix='fc'):
    if feature_learning_rate is not None:
        embedder_weights, classifier_weights = split_classfier_params(model, classifier_prefix)
        params = [dict(params=embedder_weights, lr=feature_learning_rate),
                  dict(params=classifier_weights, lr=learning_rate)]
    else:
        params = model.parameters()
    return params


def train_only(model, save_dir, train_loader, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0

        for (imgs, labels, img_names) in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs, True) # of shape 64 x 7
            loss = criterion(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        if epoch % val_interval == 0:
            test_acc, test_loss = test(model, test_loader)
            f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
            f.flush()
    print('Training done!')
    f.close()


def train_cps(model, save_dir, sup_train_loader, unsup_train_loader, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss()
    niters_per_epoch = 30
    learning_rate = 1e-4
    params_l = get_trainable_params(model.branch1, learning_rate=learning_rate)
    params_r = get_trainable_params(model.branch2, learning_rate=learning_rate)
    optimizer_l = torch.optim.Adam(params_l, lr=learning_rate, weight_decay=1e-5)
    optimizer_r = torch.optim.Adam(params_r, lr=learning_rate, weight_decay=1e-5)
    lr_lambda = lambda epoch: (1-epoch/epochs)**0.95
    scheduler_l = torch.optim.lr_scheduler.LambdaLR(optimizer_l, lr_lambda=lr_lambda)
    scheduler_r = torch.optim.lr_scheduler.LambdaLR(optimizer_r, lr_lambda=lr_lambda)
    #scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=2, gamma=0.5)
    #scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=2, gamma=0.5)
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0
        sup_trainloader = iter(sup_train_loader)
        unsup_trainloader = iter(unsup_train_loader)

        for idx in tqdm(range(niters_per_epoch)):
            sup_imgs, sup_labels, sup_img_names = next(sup_trainloader)
            unsup_imgs, _, unsup_img_names = next(unsup_trainloader)
            sup_imgs, sup_labels, unsup_imgs = sup_imgs.to(device), sup_labels.to(device), unsup_imgs.to(device)

            _, pred_sup_l = model(sup_imgs, step=1)
            _, pred_sup_r = model(sup_imgs, step=2)
            _, pred_unsup_l = model(unsup_imgs, step=1)
            _, pred_unsup_r = model(unsup_imgs, step=2)

            pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
            _, max_l = torch.max(pred_l, dim=1)
            _, max_r = torch.max(pred_r, dim=1)
            max_l = max_l.long()
            max_r = max_r.long()

            cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            sup_loss = criterion(pred_sup_l, sup_labels) + criterion(pred_sup_r, sup_labels)
            loss = cps_loss + sup_loss

            loss_item += loss.item()
            _, prediction = torch.max(pred_sup_l.data, 1)
            correct += ((prediction == sup_labels).sum()).item()
            total += len(prediction)

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            loss.backward()
            optimizer_l.step()
            optimizer_r.step()

        scheduler_l.step()
        scheduler_r.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        if epoch % val_interval == 0:
            test_acc, test_loss = test(model, test_loader)
            f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
            f.flush()
    print('Training done!')
    f.close()


def train_simple(model, save_dir, sup_train_loader, unsup_train_loader, test_loader, simple_fn, exp_args):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.to(device)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')

    niters_per_epoch = 30
    num_warmup_epochs = exp_args.num_warmup_epochs
    max_warmup_step = num_warmup_epochs * niters_per_epoch
    learning_rate = 1e-4
    
    # params = get_trainable_params(model, learning_rate=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    lambda_u = exp_args.lambda_u
    lambda_pair = exp_args.lambda_pair
    supervised_loss = build_supervised_loss(exp_args)
    unsupervised_loss = build_unsupervised_loss(exp_args)
    pair_loss = build_pair_loss(exp_args)
    ramp_up = get_ramp_up(ramp_up_type=exp_args.ramp_up_type, length=max_warmup_step)
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0
        sup_trainloader = iter(sup_train_loader)
        unsup_trainloader = iter(unsup_train_loader)
        global_step = 0
        for idx in tqdm(range(niters_per_epoch)):
            sup_imgs, sup_labels, sup_img_names = next(sup_trainloader)
            unsup_imgs, unsup_labels, unsup_img_names = next(unsup_trainloader)
            sup_imgs, sup_labels, unsup_imgs, unsup_labels = sup_imgs.to(device), sup_labels.to(device), unsup_imgs.to(device), unsup_labels.to(device)
            batch_size = len(sup_imgs)
            # == preprocess ==
            outputs = simple_fn(model=model,
                    **simple_fn.preprocess(
                        x_inputs=sup_imgs,
                        x_strong_inputs=sup_imgs,
                        x_targets=sup_labels,
                        u_inputs=unsup_imgs,
                        u_strong_inputs=unsup_imgs,
                        u_true_targets=unsup_labels)
                    )
            x_inputs=outputs['x_mixed']
            x_targets=outputs['p_mixed']
            u_inputs=outputs['u_mixed']
            u_targets=outputs['q_mixed']
            u_true_targets=outputs['q_true_mixed']
            # == compute logits ==
            batch_outputs = [x_inputs, *torch.split(u_inputs, batch_size, dim=0)]
            batch_outputs = interleave(batch_outputs, batch_size)
            batch_outputs = [model(batch_output, return_feature=False) for batch_output in batch_outputs]
            batch_outputs = interleave(batch_outputs, batch_size)
            x_logits = batch_outputs[0]
            u_logits = torch.cat(batch_outputs[1:], dim=0)
            # == compute loss ==
            x_probs = F.softmax(x_logits, dim=1)
            u_probs = F.softmax(u_logits, dim=1)
            loss_x = supervised_loss(x_logits, x_probs, x_targets)
            loss = loss_x
            ramp_up_value = ramp_up(current=global_step)
            loss_u = unsupervised_loss(u_logits, u_probs, u_targets)
            weighted_loss_u = ramp_up_value * lambda_u * loss_u
            loss += weighted_loss_u
            loss_pair = pair_loss(logits=u_logits, probs=u_probs, targets=u_targets)
            weighted_loss_pair = ramp_up_value * lambda_pair * loss_pair
            loss += weighted_loss_pair


            loss_item += loss.item()
            _, prediction = torch.max(x_logits.data, 1)
            correct += ((prediction == sup_labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
            optimizer.step()
            # model.update()
            global_step += 1

        scheduler.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        if epoch % val_interval == 0:
            test_acc, test_loss = test(model, test_loader)
            f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
            f.flush()
    print('Training done!')
    f.close()

def test(model, test_loader):
    print('Testing...')
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    loss_item = 0
    criterion = nn.CrossEntropyLoss()
    all_pred = []
    all_label = []
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            feature, res = model(imgs)  # of shape 64 x 7
            loss = criterion(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)
            for i in range(len(prediction)):
                all_pred.append(int(prediction.data.cpu()[i]))
                all_label.append(int(labels[i]))
    accuracy = correct / total
    precision = metrics.precision_score(all_label, all_pred, average='macro')
    recall = metrics.recall_score(all_label, all_pred, average='macro')
    jaccard = metrics.jaccard_score(all_label, all_pred, average='macro')
    print('Test: Acc {:.4f}, Loss {:.4f}'.format(accuracy, loss_item / total))
    print('Test: Precision: {:.4f}'.format(precision))
    print('Test: Recall: {:.4f}'.format(recall))
    print('Test: Jaccard: {:.4f}'.format(jaccard))
    return accuracy, loss_item / total


def extract(model, loader, save_path, record_err= False):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    err_dict = {}
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(loader):
            assert len(img_names) == 1 # batch_size = 1
            video, img_in_video = img_names[0].split('/')[-2], img_names[0].split('/')[-1] # video63 5730.jpg
            video_folder = os.path.join(save_path, video)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            feature_save_path = os.path.join(video_folder, img_in_video.split('.')[0] + '.npy')

            if os.path.exists(feature_save_path):
                continue
            imgs, labels = imgs.to(device), labels.to(device)
            features, res = model(imgs)

            _,  prediction = torch.max(res.data, 1)
            if record_err and (prediction == labels).sum().item() == 0:
                # hard frames
                if video not in err_dict.keys():
                    err_dict[video] = []
                else:
                    err_dict[video].append(int(img_in_video.split('.')[0]))

            features = features.to('cpu').numpy() # of shape 1 x 2048

            np.save(feature_save_path, features)

    return err_dict


def imgf2videof(source_folder, target_folder):
    '''
        Merge the extracted img feature to video feature.
    '''
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for video in os.listdir(source_folder):
        video_feature_save_path = os.path.join(target_folder, video + '.npy')
        video_abs_path = os.path.join(source_folder, video)
        nums_of_imgs = len(os.listdir(video_abs_path))
        video_feature = []
        for i in range(nums_of_imgs):
            img_abs_path = os.path.join(video_abs_path, '{:04d}.npy'.format(i+1))
            video_feature.append(np.load(img_abs_path))

        video_feature = np.concatenate(video_feature, axis=0)

        np.save(video_feature_save_path, video_feature)
        print('{} done!'.format(video))




if __name__ == '__main__':
    from wide_resnet import WideResNet
    from resnet import resnet18, resnet34
    frames_path = '../dataset/{}/frames'.format(args.dataset)
    annotations_path = '../dataset/{}/frames_annotations'.format(args.dataset)
    timestamp_path = '../dataset/{}/timestamp.npy'.format(args.dataset)
    dataset_mean = stats_dict[args.dataset]['mean']
    dataset_std = stats_dict[args.dataset]['std']
    if args.action == 'train_only':
        #inception = inception_v3(pretrained=True, aux_logits=False)
        #fc_features = inception.fc.in_features
        #inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        # net = WideResNet(in_channels=3, out_channels=7)
        net = resnet18(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))

        timestamp_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path)
        timestamp_train_dataloader = DataLoader(timestamp_traindataset, batch_size=8, shuffle=True, drop_last=False)

        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_only( net, 'models/{}/only'.format(args.dataset), timestamp_train_dataloader, full_test_dataloader)

    if args.action == 'train_cps':
        net = SemiNetwork('resnet18', len(phase2label_dicts[args.dataset]))
        unsup_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True, unsupervised=True, timestamp=timestamp_path)
        unsup_train_dataloader = DataLoader(unsup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        sup_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path)
        sup_train_dataloader = DataLoader(sup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        print('unsup dataset: {}\n sup dataset: {}\n'.format(len(unsup_traindataset), len(sup_traindataset)))
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_cps(net, 'models/{}/cps'.format(args.dataset), sup_train_dataloader, unsup_train_dataloader, full_test_dataloader)

    if args.action == 'train_simple':
        net = resnet18(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        naive_transform = transforms.Compose([
                transforms.Resize([250, 250]),
                RandomCrop(224),
                transforms.ToTensor(),
            ])
        augmenter = get_augmenter('simple', image_size=(224, 224), dataset_mean=dataset_mean, dataset_std=dataset_std)
        strong_augmenter = get_augmenter('randaugment', image_size=(224, 224), dataset_mean=dataset_mean, dataset_std=dataset_std)
        exp_args = EasyDict({
            'mixmatch_type': 'simple',
            't': 0.5,
            'k': 2,
            'k_strong': 7,
            'num_warmup_epochs': 0,
            'ramp_up_type': 'linear',
            'lambda_u': 75,
            'lambda_pair': 75,
            'u_loss_type': 'mse',
            'u_loss_thresholded': True,
            'confidence_threshold': 0.95,
            'similarity_threshold': 0.9,
            'similarity_type': 'bhc',
            'distance_loss_type': 'bhc',
            })
        simple_fn = get_mixmatch_function(exp_args, len(phase2label_dicts[args.dataset]), augmenter=augmenter, strong_augmenter=strong_augmenter)
        unsup_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True, unsupervised=True, timestamp=timestamp_path, transform=naive_transform)
        unsup_train_dataloader = DataLoader(unsup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        sup_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path, transform=naive_transform)
        sup_train_dataloader = DataLoader(sup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        print('unsup dataset: {}\n sup dataset: {}\n'.format(len(unsup_traindataset), len(sup_traindataset)))
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_simple(net, 'models/{}/simple'.format(args.dataset), sup_train_dataloader, unsup_train_dataloader, full_test_dataloader, simple_fn, exp_args)

    if args.action == 'train_mixmatch':
        net = resnet18(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        naive_transform = transforms.Compose([
                transforms.Resize([250, 250]),
                RandomCrop(224),
                transforms.ToTensor(),
            ])
        augmenter = get_augmenter('simple', image_size=(224, 224), dataset_mean=dataset_mean, dataset_std=dataset_std)
        strong_augmenter = get_augmenter('randaugment', image_size=(224, 224), dataset_mean=dataset_mean, dataset_std=dataset_std)
        exp_args = EasyDict({
            'mixmatch_type': 'mixmatch',
            't': 0.5,
            'k': 2,
            'k_strong': 7,
            'num_warmup_epochs': 0,
            'ramp_up_type': 'linear',
            'alpha': 0.75,
            'lambda_u': 50,
            'lambda_pair': 50,
            'u_loss_type': 'mse',
            'u_loss_thresholded': True,
            'confidence_threshold': 0.95,
            'similarity_threshold': 0.9,
            'similarity_type': 'bhc',
            'distance_loss_type': 'bhc',
            })
        simple_fn = get_mixmatch_function(exp_args, len(phase2label_dicts[args.dataset]), augmenter=augmenter, strong_augmenter=strong_augmenter)
        unsup_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True, unsupervised=True, timestamp=timestamp_path, transform=naive_transform)
        unsup_train_dataloader = DataLoader(unsup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        sup_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path, transform=naive_transform)
        sup_train_dataloader = DataLoader(sup_traindataset, batch_size=8, shuffle=True, drop_last=False)
        print('unsup dataset: {}\n sup dataset: {}\n'.format(len(unsup_traindataset), len(sup_traindataset)))
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_simple(net, 'models/{}/mixmatch'.format(args.dataset), sup_train_dataloader, unsup_train_dataloader, full_test_dataloader, simple_fn, exp_args)


    if args.action == 'extract_only': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/only/5.model'.format(args.dataset)
        inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@only/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@only/'.format(args.dataset), '{}/train_dataset/video_feature@only/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@only/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@only/'.format(args.dataset), '{}/test_dataset/video_feature@only/'.format(args.dataset))
    
    if args.action == 'extract_cps': # extract inception feature
        net = SemiNetwork(len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/cps/7.model'.format(args.dataset)
        net.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(net, full_train_dataloader, '{}/train_dataset/frame_feature@cps/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@cps/'.format(args.dataset), '{}/train_dataset/video_feature@cps/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(net, full_test_dataloader, '{}/test_dataset/frame_feature@cps/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@cps/'.format(args.dataset), '{}/test_dataset/video_feature@cps/'.format(args.dataset))
    
    if args.action == 'extract_simple': # extract inception feature
        net = resnet18(pretrained=True)
        fc_features = net.fc.in_features
        net.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/simple/10.model'.format(args.dataset)
        net.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(net, full_train_dataloader, '{}/train_dataset/frame_feature@simple/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@simple/'.format(args.dataset), '{}/train_dataset/video_feature@simple/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(net, full_test_dataloader, '{}/test_dataset/frame_feature@simple/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@simple/'.format(args.dataset), '{}/test_dataset/video_feature@simple/'.format(args.dataset))

