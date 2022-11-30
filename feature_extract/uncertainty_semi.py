import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

import os
import time
import argparse
import numpy as np
import random
from tqdm import tqdm

from data_util import base_transform, phase2label_dicts, PureTimestampDataset, PseudoLabelDataset, FullDataset, BatchGenerator
from model import inception_v3, SemiNetwork
from simclr import ContrastiveLearningViewGenerator, SimCLR
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
parser.add_argument('--action')
parser.add_argument('--dataset', default="cholec80", choices=['cholec80','m2cai16'])
parser.add_argument('--target', type=str, default='train_set')
parser.add_argument('--pseudo', action='store_true')
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--arch', type=str)
args = parser.parse_args()

epochs = args.num_epochs


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def class_balanced_loss(num_classes, logit, target, beta=0.99):
    samples_per_cls = []
    for l in range(num_classes):
        samples_per_cls.append(int(torch.sum(target==l)))
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = torch.tensor(weights).float().to(logit.device)
    prob = logit.log_softmax(dim=1)
    cb_loss = F.cross_entropy(input=prob, target=target, weight=weights)
    return cb_loss

def train_only(model, save_dir, train_loader, test_loader):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    scaler = torch.cuda.amp.GradScaler()
    model.to(device)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()

        correct = 0
        total = 0
        loss_item = 0

        for (imgs, labels, img_names) in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                feature, res = model(imgs) # of shape 64 x 7
                loss = criterion(res, labels)
            loss_item += loss.item()
            _, prediction = torch.max(res.data, 1)
            correct += ((prediction == labels).sum()).item()
            total += len(prediction)

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
        f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
        f.flush()
        torch.save(model.state_dict(), save_dir + "/{}.model".format(epoch))
        #test_acc, test_loss = test(model, test_loader)
        #f.write('Test Acc: {}, Loss: {}'.format(test_acc, test_loss) + '\n')
        f.flush()
        test_acc, test_loss = test(model, test_loader)
        f.write('Test Acc: {:.4f}, Loss: {:.4f}'.format(test_acc, test_loss) + '\n')
        f.flush()
    print('Training done!')


    f.close()


def train_semi(save_dir, batch_gen, test_loader, num_iters, forward_times):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    for it in range(1, num_iters+1):
        model = SemiNetwork('inception_v3', len(phase2label_dicts[args.dataset]))
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        learning_rate = 1e-4
        optimizer_l = torch.optim.Adam(model.branch1.parameters(), learning_rate, weight_decay=1e-5)
        optimizer_r = torch.optim.Adam(model.branch2.parameters(), learning_rate, weight_decay=1e-5)
        scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_l, step_size=2, gamma=0.5)
        scheduler_r = torch.optim.lr_scheduler.StepLR(optimizer_r, step_size=2, gamma=0.5)
        print('Iteration: {}. Num of pseudo labels: {}, rate: {:.4f}'.format(it, len(batch_gen.imgs), len(batch_gen.imgs) / len(batch_gen.total_imgs)))
        print('Num of unlabeled data: {}'.format(len(batch_gen.unsup_imgs)))
        for epoch in range(1, epochs + 1):
            model.train()

            correct = 0
            total = 0
            loss_item = 0

            while batch_gen.has_next() and batch_gen.has_next_unsup():
                sup_imgs, sup_labels = batch_gen.next_batch()
                unsup_imgs = batch_gen.next_batch_unsup()
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
                if it > 3 and epoch > 2:
                    loss = cps_loss + sup_loss
                else:
                    loss = sup_loss

                loss_item += loss.item()
                _, prediction = torch.max(pred_sup_l.data, 1)
                correct += ((prediction == sup_labels).sum()).item()
                total += len(prediction)

                optimizer_l.zero_grad()
                optimizer_r.zero_grad()
                loss.backward()
                optimizer_l.step()
                optimizer_r.step()
            batch_gen.reset()
            batch_gen.reset_unsup()

            scheduler_l.step()
            scheduler_r.step()

            print('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total))
            f.write('Train Epoch {}: Acc {}, Loss {}'.format(epoch, correct/total, loss_item/total) + '\n')
            f.flush()
            torch.save(model.state_dict(), save_dir + "/{}_{}.model".format(it, epoch))

        test_acc, test_loss = test(model, test_loader)
        f.write('Test Acc: {:.4f}, Loss: {:.4f}'.format(test_acc, test_loss) + '\n')
        f.flush()

        start = time.time()
        model.eval()
        enable_dropout(model)
        all_predictions = []
        all_scores = []
        with torch.no_grad():
            '''
            while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                mc_probs = []
                for _ in range(forward_times):
                    _, logits = model(data)
                    probs = F.softmax(logits, dim=1)
                    mc_probs.append(probs)
                mc_probs = torch.stack(mc_probs, dim=0)
                std = torch.std(mc_probs, dim=0)
                mc_probs = torch.mean(mc_probs, dim=0)
                predictions = torch.max(mc_probs, dim=1)[1]
                scores = std.gather(1, predictions.unsqueeze(-1))
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(scores.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()
            '''
            while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                _, logits = model(data)
                probs = F.softmax(logits, dim=1)
                predictions = torch.max(probs, dim=1)[1]
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(probs.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()
        video_lengths = batch_gen.video_lengths
        assert len(all_predictions) == sum(video_lengths)
        pre = 0
        new_all_predictions = []
        new_all_scores = []
        for length in video_lengths:
            new_all_predictions.append(all_predictions[pre: pre+length])
            new_all_scores.append(all_scores[pre: pre+length])
            pre += length
        thres = 0.95
        batch_gen.update_dataset(new_all_predictions, new_all_scores, thres=thres)
        batch_gen.reset()
        print('Pseudo labeling time: {}s'.format(time.time() - start))

    print('Training done!')
    f.close()


def train_uncertainty(save_dir, batch_gen, test_loader, num_iters, forward_times):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    num_classes = 7 if args.dataset == 'cholec80' else 8
    for it in range(1, num_iters+1):
        num_per_class = np.zeros(num_classes)
        model = inception_v3(pretrained=True, aux_logits=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model.to(device)
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        print('Iteration: {}. Num of pseudo labels: {}, rate: {:.4f}'.format(it, len(batch_gen.imgs), len(batch_gen.imgs) / len(batch_gen.total_imgs)))
        start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            correct = 0
            total = 0
            loss_item = 0
            while batch_gen.has_next():
                data, target = batch_gen.next_batch()
                data, target = data.to(device), target.to(device)
                _, logits = model(data)

                loss = criterion(logits, target)
                # loss = class_balanced_loss(num_classes, logits, target)

                loss_item += loss.item()
                _, prediction = torch.max(logits.data, 1)
                correct += ((prediction == target).sum()).item()
                total += len(prediction)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for c in range(num_classes):
                    num_per_class[c] += (target==c).sum()
            scheduler.step()
            batch_gen.reset()

            print('Train Epoch {}: Acc {:.4f}, Loss {:.4f}'.format(epoch, correct/total, loss_item/total))
            f.write('Train Epoch {}: Acc {:.4f}, Loss {:.4f}'.format(epoch, correct/total, loss_item/total) + '\n')
            f.flush()
            torch.save(model.state_dict(), save_dir + "/{}_{}.model".format(it, epoch))
        print('Training time: {}s'.format(time.time() - start))
        print('Number of per class:', (num_per_class / epoch).tolist())

        start = time.time()
        test_acc, test_loss = test(model, test_loader)
        f.write('Test Acc: {:.4f}, Loss: {:.4f}'.format(test_acc, test_loss) + '\n')
        f.flush()
        print('Testing time: {}s'.format(time.time() - start))
        
        start = time.time()
        model.eval()
        enable_dropout(model)
        all_predictions = []
        all_scores = []
        with torch.no_grad():
            '''
            while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                mc_probs = []
                for _ in range(forward_times):
                    _, logits = model(data)
                    probs = F.softmax(logits, dim=1)
                    mc_probs.append(probs)
                mc_probs = torch.stack(mc_probs, dim=0)
                std = torch.std(mc_probs, dim=0)
                mc_probs = torch.mean(mc_probs, dim=0)
                predictions = torch.max(mc_probs, dim=1)[1]
                scores = std.gather(1, predictions.unsqueeze(-1))
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(scores.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()
            '''
            while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                _, logits = model(data)
                probs = F.softmax(logits, dim=1)
                predictions = torch.max(probs, dim=1)[1]
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(probs.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()
        video_lengths = batch_gen.video_lengths
        assert len(all_predictions) == sum(video_lengths)
        pre = 0
        new_all_predictions = []
        new_all_scores = []
        for length in video_lengths:
            new_all_predictions.append(all_predictions[pre: pre+length])
            new_all_scores.append(all_scores[pre: pre+length])
            pre += length
        thres = 0.95
        batch_gen.update_dataset(new_all_predictions, new_all_scores, thres=thres)
        batch_gen.reset()
        print('Pseudo labeling time: {}s'.format(time.time() - start))
    print('Training done!')
    f.close()

def train_temporal(save_dir, batch_gen, test_loader, num_iters, forward_times, seg_length=8):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'log.txt'), 'w')
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    mse = nn.MSELoss(reduction='none')
    num_classes = 7 if args.dataset == 'cholec80' else 8
    for it in range(1, num_iters+1):
        num_per_class = np.zeros(num_classes)
        model = inception_v3(pretrained=True, aux_logits=False)
        fc_features = model.fc.in_features
        model.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model.to(device)
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        print('Iteration: {}. Num of pseudo labels: {}, rate: {:.4f}'.format(it, len(batch_gen.imgs), len(batch_gen.imgs) / len(batch_gen.total_imgs)))
        start = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            correct = 0
            total = 0
            loss_item = 0
            while batch_gen.has_next() and batch_gen.has_next_seg():
                data, target = batch_gen.next_batch()
                data, target = data.to(device), target.to(device)
                _, logits = model(data)

                loss = criterion(logits, target)
                if it >= 3:
                    seg_data = batch_gen.next_batch_seg()
                    seg_data = seg_data.to(device)
                    _, seg_logits = model(seg_data)
                    seg_logits = seg_logits.reshape(-1, seg_length, seg_logits.size(-1))
                    loss += 0.015 * torch.mean(torch.clamp(
                        mse(F.log_softmax(seg_logits[:, 1:, :], dim=1), F.log_softmax(seg_logits.detach()[:, :-1, :], dim=1)),
                        min=0, max=16
                        ))
                # loss = class_balanced_loss(num_classes, logits, target)

                loss_item += loss.item()
                _, prediction = torch.max(logits.data, 1)
                correct += ((prediction == target).sum()).item()
                total += len(prediction)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for c in range(num_classes):
                    num_per_class[c] += (target==c).sum()
            scheduler.step()
            batch_gen.reset()
            batch_gen.reset_seg()

            print('Train Epoch {}: Acc {:.4f}, Loss {:.4f}'.format(epoch, correct/total, loss_item/total))
            f.write('Train Epoch {}: Acc {:.4f}, Loss {:.4f}'.format(epoch, correct/total, loss_item/total) + '\n')
            f.flush()
            torch.save(model.state_dict(), save_dir + "/u_{}_{}.model".format(it, epoch))
        print('Training time: {}s'.format(time.time() - start))
        print('Number of per class:', (num_per_class / epoch).tolist())

        start = time.time()
        test_acc, test_loss = test(model, test_loader)
        f.write('Test Acc: {:.4f}, Loss: {:.4f}'.format(test_acc, test_loss) + '\n')
        f.flush()
        print('Testing time: {}s'.format(time.time() - start))
        
        start = time.time()
        model.eval()
        enable_dropout(model)
        all_predictions = []
        all_scores = []
        with torch.no_grad():
            
            while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                mc_probs = []
                for _ in range(forward_times):
                    _, logits = model(data)
                    probs = F.softmax(logits, dim=1)
                    mc_probs.append(probs)
                mc_probs = torch.stack(mc_probs, dim=0)
                std = torch.std(mc_probs, dim=0)
                mc_probs = torch.mean(mc_probs, dim=0)
                predictions = torch.max(mc_probs, dim=1)[1]
                scores = std.gather(1, predictions.unsqueeze(-1))
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(scores.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()
            
            '''while batch_gen.has_next_total():
                data = batch_gen.next_batch_total()
                data = data.to(device)
                _, logits = model(data)
                probs = F.softmax(logits, dim=1)
                predictions = torch.max(probs, dim=1)[1]
                all_predictions.extend(predictions.detach().cpu().numpy().tolist())
                all_scores.extend(probs.view(-1).detach().cpu().numpy().tolist())
            batch_gen.reset_total()'''
        video_lengths = batch_gen.video_lengths
        assert len(all_predictions) == sum(video_lengths)
        pre = 0
        new_all_predictions = []
        new_all_scores = []
        for length in video_lengths:
            new_all_predictions.append(all_predictions[pre: pre+length])
            new_all_scores.append(all_scores[pre: pre+length])
            pre += length
        thres = 0.1
        batch_gen.update_dataset(new_all_predictions, new_all_scores, thres=thres)
        batch_gen.reset()
        print('Pseudo labeling time: {}s'.format(time.time() - start))
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
    print('Test: Acc {:.4f}, Loss {:.4f}'.format(accuracy, loss_item / total))
    print('Test: Precision: {:.4f}'.format(precision))
    print('Test: Recall: {:.4f}'.format(recall))
    return accuracy, loss_item / total


def extract(model, loader, save_path, record_err=False):
    model.eval()
    model.to(device)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # err_dict = {}
    with torch.no_grad():
        for (imgs, labels, img_names) in tqdm(loader):
            videos, img_in_videos = [img_name.split('/')[-2] for img_name in img_names],\
                    [img_name.split('/')[-1] for img_name in img_names] # video63 5730.jpg
            video_folders = [os.path.join(save_path, video) for video in videos]
            for video_folder in video_folders:
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

            #if os.path.exists(feature_save_path):
            #    continue
            imgs, labels = imgs.to(device), labels.to(device)
            features, res = model(imgs)

            #_,  prediction = torch.max(res.data, 1)
            #if record_err and (prediction == labels).sum().item() == 0:
            #    # hard frames
            #    if video not in err_dict.keys():
            #        err_dict[video] = []
            #    else:
            #        err_dict[video].append(int(img_in_video.split('.')[0]))
            for video_folder, img_in_video, feature in zip(video_folders, img_in_videos, features):
                feature = feature.unsqueeze(0).cpu().numpy() # of shape 1 x 2048
                feature_save_path = os.path.join(video_folder, img_in_video.split('.')[0]+'.npy')
                np.save(feature_save_path, feature)

    # return err_dict

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
        # print('{} done!'.format(video))

if __name__ == '__main__':

    frames_path = '../dataset/{}/frames'.format(args.dataset)
    annotations_path = '../dataset/{}/frames_annotations'.format(args.dataset)
    frames_25pfs_path = '../dataset/{}/frames_25fps'.format(args.dataset)
    annotations_25fps_path = '../dataset/{}/phase_annotations'.format(args.dataset)
    timestamp_path = '../dataset/{}/timestamp.npy'.format(args.dataset)
    pseudo_label_path = '../models/{}/arch-{}_extract-only_pseudo-confidence/pseudo_labels.npy'.format(args.dataset, args.arch)
    if args.action == 'train_only':
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))

        timestamp_traindataset = PureTimestampDataset(args.dataset, frames_path, timestamp_path)
        timestamp_train_dataloader = DataLoader(timestamp_traindataset, batch_size=8, shuffle=True, drop_last=False)

        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_only( inception, 'models/{}/only'.format(args.dataset), timestamp_train_dataloader, full_test_dataloader)

    if args.action == 'train_simclr':
        model = inception_v3(pretrained=True, aux_logits=False).to(device)
        dataset = FullDataset(args.dataset, frames_path, annotations_path, train=True, transform=ContrastiveLearningViewGenerator(base_transform(args.dataset)))
        train_loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, drop_last=True)
        args.batch_size = 8
        args.n_views = 2
        args.device = device
        args.epochs = epochs
        args.temperature = 0.07
        args.fp16_precision = True
        optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train('models/{}/simclr'.format(args.dataset), train_loader)

    if args.action == 'train_semi':
        batch_gen = BatchGenerator(args.dataset, frames_path, annotations_path, timestamp_path, batch_size=8, sample_rate=5)
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=8, shuffle=True, drop_last=False)

        train_semi('models/{}/semi'.format(args.dataset), batch_gen, full_test_dataloader, num_iters=5, forward_times=5)

    if args.action == 'train_uncertainty':
        batch_gen = BatchGenerator(args.dataset, frames_path, annotations_path, timestamp_path, batch_size=8)
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_uncertainty('models/{}/uncertainty'.format(args.dataset), batch_gen, full_test_dataloader, num_iters=15, forward_times=5)

    if args.action == 'train_temporal':
        batch_gen = BatchGenerator(args.dataset, frames_path, annotations_path, timestamp_path, batch_size=8)
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_temporal('models/{}/temporal'.format(args.dataset), batch_gen, full_test_dataloader, num_iters=15, forward_times=5)


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

    if args.action == 'extract_simclr':
        inception = inception_v3(pretrained=True, aux_logits=False)
        model_path = 'models/{}/simclr/1.model'.format(args.dataset)
        inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@simclr/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@simclr/'.format(args.dataset), '{}/train_dataset/video_feature@simclr/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@simclr/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@simclr/'.format(args.dataset), '{}/test_dataset/video_feature@simclr/'.format(args.dataset))

    
    if args.action == 'extract_semi': # extract inception feature
        net = SemiNetwork('inception_v3', len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/semi/5.model'.format(args.dataset)
        net.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=64, shuffle=False, drop_last=False)
            extract(net, full_train_dataloader, '{}/train_dataset/frame_feature@semi/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@semi/'.format(args.dataset), '{}/train_dataset/video_feature@semi/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=False, drop_last=False)

            extract(net, full_test_dataloader, '{}/test_dataset/frame_feature@semi/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@semi/'.format(args.dataset), '{}/test_dataset/video_feature@semi/'.format(args.dataset))

    if args.action == 'extract_uncertainty': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/uncertainty/10_5.model'.format(args.dataset)
        inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=1, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@uncertainty/'.format(args.dataset))
            imgf2videof('{}/train_dataset/frame_feature@uncertainty/'.format(args.dataset), '{}/train_dataset/video_feature@uncertainty/'.format(args.dataset))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=1, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@uncertainty/'.format(args.dataset))
            imgf2videof('{}/test_dataset/frame_feature@uncertainty/'.format(args.dataset), '{}/test_dataset/video_feature@uncertainty/'.format(args.dataset))
    
    if args.action == 'test':
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/uncertainty/11_5.model'.format(args.dataset)
        inception.load_state_dict(torch.load(model_path))
        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        test(inception, full_test_dataloader)

    if args.action == 'extract_pseudo': # extract inception feature
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))
        model_path = 'models/{}/confidence_pseudo_{}/5.model'.format(args.dataset, args.arch)
        inception.load_state_dict(torch.load(model_path))

        if args.target == 'train_set':
            full_traindataset = FullDataset(args.dataset, frames_path, annotations_path, train=True)
            full_train_dataloader = DataLoader(full_traindataset, batch_size=16, shuffle=False, drop_last=False)
            extract(inception, full_train_dataloader, '{}/train_dataset/frame_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch))
            imgf2videof('{}/train_dataset/frame_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch), '{}/train_dataset/video_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch))
        else:
            full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False)
            full_test_dataloader = DataLoader(full_testdataset, batch_size=16, shuffle=False, drop_last=False)

            extract(inception, full_test_dataloader, '{}/test_dataset/frame_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch))
            imgf2videof('{}/test_dataset/frame_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch), '{}/test_dataset/video_feature@confidence_pseudo_{}/'.format(args.dataset, args.arch))
 
    if args.action == 'train_pseudo':
        inception = inception_v3(pretrained=True, aux_logits=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, len(phase2label_dicts[args.dataset]))

        if args.pseudo:
            timestamp_traindataset = PseudoLabelDataset(args.dataset, frames_path, annotations_path, pseudo_label_path, sample_rate=1)
            timestamp_train_dataloader = DataLoader(timestamp_traindataset, batch_size=8, shuffle=True, drop_last=False)
        else:
            timestamp_traindataset = PureTimestampDataset(args.dataset, frames_path, annotations_path, timestamp_path)
            timestamp_train_dataloader = DataLoader(timestamp_traindataset, batch_size=8, shuffle=True, drop_last=False)

        full_testdataset = FullDataset(args.dataset, frames_path, annotations_path, train=False, sample_rate=5)
        full_test_dataloader = DataLoader(full_testdataset, batch_size=64, shuffle=True, drop_last=False)

        train_only(inception, 'models/{}/confidence_pseudo_{}'.format(args.dataset, args.arch), timestamp_train_dataloader, full_test_dataloader)
