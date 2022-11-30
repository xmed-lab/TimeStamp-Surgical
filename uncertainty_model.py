import os
import random
from tabulate import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.swa_utils import AveragedModel, SWALR
import copy
from copy import deepcopy
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, arch):
        super(MultiStageModel, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes, arch)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3, arch))
                                     for s in range(num_stages-1)])

    def forward(self, x, mask):
        middle_out, out = self.tower_stage(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return middle_out, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, arch):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3, arch)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5, arch)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size, arch):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        if arch == 'casual':
            self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualCasualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                        for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                        for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class DilatedResidualCasualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualCasualLayer, self).__init__()
        self.padding = 2 * int(dilation + dilation * (kernel_size -3) / 2)
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.pad(x, [self.padding, 0], 'constant', 0)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, test_features, test_gt_path, phase2label, device, num_blocks, num_layers, num_f_maps, dim, num_classes, args):
        self.test_features = test_features
        self.test_gt_path = test_gt_path
        self.phase2label = phase2label
        self.device = device
        #self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes, args.arch)
        self.arch = args.arch
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps
        self.dim = dim
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.kl = nn.KLDivLoss(reduction='none')
        self.num_classes = num_classes

        self.pseudo = args.pseudo
        self.num_epochs = args.num_epochs
        self.noisy = args.noisy
        self.uncertainty_warmup_epochs = args.uncertainty_warmup_epochs
        self.max_thres = args.max_thres
        self.smooth = args.smooth
        self.lambda_smooth = args.lambda_smooth
        self.entropy = args.entropy
        self.lambda_entropy = args.lambda_entropy
        self.forward_times = args.forward_times
        self.beta = 0.99

        self.visualization = args.visualization

    def adjust_uncertainty_thres(self, epoch):
        if self.num_classes == 7:  # cholec80
            self.thres = 0.1
        else:
            self.thres = 0.1

    def confidence_loss(self, pred, confidence_mask):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(self.device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss

    def class_balanced_loss(self, logit, target):
        samples_per_cls = []
        for l in range(self.num_classes):
            samples_per_cls.append(int(torch.sum(target==l)))
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = torch.tensor(weights).float().to(self.device)
        prob = logit.log_softmax(dim=1)
        cb_loss = F.cross_entropy(input=prob, target=target, weight=weights, ignore_index=-100)
        return cb_loss

    def uncertainty_loss(self, logits, thres):
        prob = F.softmax(logits.transpose(2, 1), dim=1)
        entropy = torch.sum(- prob * torch.log(prob), dim=2)
        mask = entropy < thres
        return (entropy * mask).mean()

    def uncertainty_loss_star(self, logits, pseudo_label):
        prob = F.softmax(logits.transpose(2, 1), dim=2)
        entropy = torch.sum(- prob * torch.log(prob), dim=2)
        mask = (pseudo_label != -100)
        return (entropy*entropy * mask).sum() / mask.sum()

    def uncertainty_loss_score(self, logits, scores):
        prob = F.softmax(logits.transpose(2, 1), dim=2)
        entropy = torch.sum(- prob * torch.log(prob), dim=2)
        scores = torch.sigmoid(scores)
        return (entropy * scores).sum() / scores.sum()

    def entropy_regularization(self, logits, mask):
        prob = F.softmax(logits, dim=1)
        entropy = torch.sum(-prob * torch.log(prob), dim=1)
        entropy = entropy * mask
        return torch.sum(entropy) / torch.sum(mask)

    def multi_train(self, save_dir, batch_gen, batch_size, learning_rate):
        # self.train(save_dir, batch_gen, batch_size, learning_rate)
        if self.num_classes == 7:
            times = 3
        else:
            times = 6
        for i in range(times):
            self._multi_train(save_dir, batch_gen, batch_size, learning_rate, i)

    def label_dropout(self, batch_pseudo_target, p):
        pseudo_target = batch_pseudo_target.clone()
        for b in range(pseudo_target.size(0)):
            for f in range(pseudo_target.size(1)):
                if pseudo_target[b, f] != -100:
                    r = random.random()
                    pseudo_target[b, f] = -100 if r<=p else pseudo_target[b, f]
        return pseudo_target

    def mask_boundary(self, batch_target):
        ret_target = deepcopy(batch_target)
        for i in range(batch_target.size(0)):
            boundaries = []
            for j in range(batch_target.size(1)-1):
                if batch_target[i, j] != batch_target[i, j+1]:
                    boundaries.append(j+1)
            for b in boundaries:
                ret_target[i, b-10:b+10] = -100
        return ret_target

    def add_noisy_labels(self, pseudo_target, timestamps, seq_lengths, noisy_length=10, C=0.5):
        from scipy import stats
        norm = stats.norm(0, 1)
        pseudo_target = pseudo_target.clone()
        for b in range(len(pseudo_target)):
            timestamp = timestamps[b]
            seq_length = seq_lengths[b]
            for i in range(len(timestamp)):
                pos = timestamp[i]
                L = 0 if i==0 else timestamp[i-1]+1
                R = seq_length-1 if i==len(timestamp)-1 else timestamp[i+1]-1
                left = min(max(0, pos-L-noisy_length), noisy_length)
                right = min(max(R-pos-noisy_length, 1), noisy_length)
                positions = np.array(list(range(pos-left, pos)) + list(range(pos+1, pos+right)))
                if len(positions) == 0:
                    continue
                probabilites = np.array([norm.pdf(np.abs(p - pos) / noisy_length) for p in positions])
                probabilites = probabilites / np.sum(probabilites)
                choice = np.random.choice(positions, int(len(positions)*C), p=probabilites, replace=False)
                pseudo_target[b][choice.tolist()] = int(pseudo_target[b][timestamp[i]])
        return pseudo_target

    def _multi_train(self, save_dir, batch_gen, batch_size, learning_rate, iter_times):
        self.model = MultiStageModel(self.num_blocks, self.num_layers, self.num_f_maps, self.dim, self.num_classes, self.arch)
        self.model.to(self.device)
        swa_model = AveragedModel(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        swa_start = 45
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
        #scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=self.num_epochs, cycle_mult=1.0, max_lr=learning_rate, min_lr=1e-5, warmup_steps=5, gamma=1.0)
        writer = SummaryWriter(save_dir)
        total_lengths = 0
        labeled_lengths = 0
        correct_lengths = 0
        self.adjust_uncertainty_thres(iter_times)
        for epoch in range(1, self.num_epochs+1):
            #if iter_times == 0:
            #    batch_gen.update_pseudo_labels_noisy()
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            if self.pseudo == 'confidence' and epoch == self.num_epochs:
                confidence_labels = {}
            while batch_gen.has_next():
                batch_input, batch_target, batch_pseudo_target, mask, batch_confidence = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_pseudo_target, mask = batch_input.to(self.device), batch_target.to(self.device),\
                        batch_pseudo_target.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                middle_pred, predictions = self.model(batch_input, mask)
                timestamp_mask = batch_gen.get_single_random(batch_input.size(-1)).to(self.device)
                vids, seq_lengths, timestamps = batch_gen.get_useful_info()
                # batch_pseudo_target = self.label_dropout(batch_pseudo_target, 0.5)
                if self.noisy:
                    batch_pseudo_target = self.add_noisy_labels(batch_pseudo_target, timestamps, seq_lengths)
                if epoch == self.num_epochs:
                    total_lengths += sum(seq_lengths)
                    labeled_lengths += torch.sum(batch_pseudo_target != -100)
                    for b, seq_length in enumerate(seq_lengths):
                        correct_lengths += torch.sum(batch_pseudo_target[b, :seq_length] == batch_target[b, :seq_length])
                loss = 0
                if self.visualization and self.pseudo == 'confidence' and epoch == self.num_epochs:
                    pseudo_labels = batch_gen.get_boundary(batch_input.size(0), middle_pred)
                    vis_dir = os.path.join(save_dir, 'train_vis')
                    if not os.path.exists(vis_dir):
                        os.makedirs(vis_dir)
                    for i, vi in enumerate(vids):
                        pseudo_label = pseudo_labels[i].squeeze(0)
                        gt = batch_target[i].squeeze(0)
                        length = seq_lengths[i]
                        predicted = torch.max(predictions[-1].data, 1)[1][i].squeeze(0)
                        self.segment_bars(os.path.join(vis_dir, '{}.png'.format(vi)), ['gt', gt[:length]], ['pseudo label', pseudo_label[:length]], ['predict', predicted[:length]])
                if self.pseudo == 'confidence' and epoch == self.num_epochs:
                    pseudo_labels = batch_gen.get_boundary(batch_input.size(0), middle_pred)
                    for i, vi in enumerate(vids):
                        confidence_labels[vi] = pseudo_labels[i].squeeze(0)[:seq_lengths[i]]
                for pi, p in enumerate(predictions):
                    if self.pseudo == 'uncertainty' or self.pseudo == 'self':
                        target = batch_pseudo_target
                    elif self.pseudo == 'uniform':
                        target = batch_gen.get_average(batch_input.size(2)).to(self.device)
                    elif self.pseudo == 'confidence':
                        if epoch > 30:
                            target = batch_gen.get_boundary(batch_input.size(0), middle_pred.detach()).to(self.device)
                        else:
                            target = batch_pseudo_target
                    else:
                        target = batch_target
                        # target = self.mask_boundary(target)
                    ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), target.view(-1))
                    loss += ce_loss
                    #cb_loss = self.class_balanced_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes), target.view(-1))
                    #loss += cb_loss
                    #print('ce', cs_loss)
                    if self.smooth:
                        smooth_loss = self.lambda_smooth * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16) * mask[:, :, 1:])
                        #smooth_loss = self.lambda_smooth * torch.mean(
                        #    self.kl(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)) * mask[:, :, 1:])
                        loss += smooth_loss
                    if self.pseudo == 'confidence' and epoch > 30:
                        confidence_loss = 0.075 * self.confidence_loss(p, batch_confidence)
                        loss += confidence_loss
                    if self.entropy:
                        labeled_mask = target.view(-1) != -100
                        entropy_loss = 0.1 * self.entropy_regularization(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labeled_mask)
                        loss += entropy_loss
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            if self.pseudo == 'confidence' and epoch == self.num_epochs:
                np.save(os.path.join(save_dir, 'pseudo_labels.npy'), confidence_labels, allow_pickle=True)
            if epoch > swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
            else:
                scheduler.step()
            batch_gen.reset()
            print('Iter {} Epoch {} training acc :{}'.format(iter_times, epoch, correct / total))

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch) + ".opt")
            # self.predict(save_dir, epoch, batch_gen.sample_rate)
        print('Pseudo labeling rate: {:.4f}, pseudo labels accuracy: {:.4f}'.format(labeled_lengths / total_lengths, correct_lengths / labeled_lengths))
        print('Uncertainty threshold: {:.4f}'.format(self.thres))
        self.predict(save_dir, self.num_epochs, batch_gen.sample_rate)
        self.model.eval()
        '''if self.pseudo == 'self':
            with torch.no_grad():
                while batch_gen.has_next():
                    batch_input, _, _, mask = batch_gen.next_batch(batch_size)
                    batch_input, mask = batch_input.to(self.device), mask.to(self.device)
                    _, logits = self.model(batch_input, mask)
                    batch_gen.update_self_labels(logits[-1].detach().transpose(2,1))
                batch_gen.reset()
            return'''
        enable_dropout(self.model)
        with torch.no_grad():
            while batch_gen.has_next():
                batch_input, _, _, mask, _ = batch_gen.next_batch(batch_size)
                batch_input, mask = batch_input.to(self.device), mask.to(self.device)
                mc_probs = []
                for _ in range(self.forward_times):
                    _, logits = self.model(batch_input, mask)
                    probs = F.softmax(logits[-1].squeeze(0).detach().transpose(2, 1), dim=2)
                    mc_probs.append(probs)
                mc_probs = torch.stack(mc_probs, dim=0)
                std = torch.std(mc_probs, dim=0)
                mc_probs = torch.mean(mc_probs, dim=0)
                predictions = torch.max(mc_probs, dim=2)[1]
                uncertainty_scores = std.gather(2, predictions.unsqueeze(-1))
                # uncertainty_scores = torch.sum(-mc_probs * torch.log(mc_probs), dim=2)
                batch_gen.update_pseudo_labels(predictions, uncertainty_scores, thres=self.thres)
                #batch_gen.update_pseudo_labels_all(predictions, uncertainty_scores, thres=self.thres)
                if False and self.visualization:
                    data_dir = os.path.join(save_dir, 'train_epoch{}_score'.format(epoch))
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    vids, seq_lengths, timestamp_pos = batch_gen.get_useful_info()
                    for b, score in enumerate(uncertainty_scores):
                        np.save(os.path.join(data_dir, '{}.npy'.format(vids[b])), {'timestamp': timestamp_pos[b], 'score': score[:seq_lengths[b]].cpu().numpy()})
            batch_gen.reset()

            if self.visualization:
                self.visualize(batch_gen, save_dir, iter_times, batch_size)
        batch_gen.save_pseudo_labels(save_dir, iter_times)

    def train(self, save_dir, batch_gen, batch_size, learning_rate):
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)
        #scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=self.num_epochs, cycle_mult=1.0, max_lr=learning_rate, min_lr=1e-5, warmup_steps=5, gamma=1.0)
        writer = SummaryWriter(save_dir)
        for epoch in range(1, self.num_epochs+1):
            self.adjust_uncertainty_thres(epoch)
            lr = scheduler.get_lr()[0]
            thres = self.thres
            print('='*20 + 'Epoch {}'.format(epoch) + '='*20)
            print('learning rate = {:.4f}, uncertainty thresold = {:.4f}'.format(lr, thres))
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, batch_pseudo_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, batch_pseudo_target, mask = batch_input.to(self.device), batch_target.to(self.device),\
                        batch_pseudo_target.to(self.device), mask.to(self.device)
                optimizer.zero_grad()
                _, predictions = self.model(batch_input, mask)
                timestamp_mask = batch_gen.get_single_random(batch_input.size(-1)).to(self.device)
                _, seq_lengths, _ = batch_gen.get_useful_info()
                total_lengths = sum(seq_lengths)
                labeled_lengths = torch.sum(batch_pseudo_target != -100)
                loss = 0
                for pi, p in enumerate(predictions):
                    if self.pseudo == 'full':
                        target = batch_target
                    else:
                        target = batch_pseudo_target
                    #ce_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), target.view(-1))
                    #loss += ce_loss
                    cb_loss = self.class_balanced_loss(p.transpose(2, 1).contiguous().view(-1, self.num_classes), target.view(-1))
                    loss += cb_loss
                    #print('ce', cs_loss)
                    if self.smooth:
                        smooth_loss = self.lambda_smooth * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16) * mask[:, :, 1:])
                        loss += smooth_loss

                    if self.entropy:
                        entropy_loss = 0.1 * self.entropy_regularization(p.transpose(2, 1).contiguous().view(-1, self.num_classes))
                        loss += entropy_loss
                #if epoch >= self.start_epoch:
                    #print('ce loss: {:.4f}, smooth_loss: {:.4f}, uncertainty_loss: {:.4f}'.format(ce_loss, smooth_loss, uncertainty_loss))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
            scheduler.step()
            batch_gen.reset()

            if self.pseudo == 'uncertainty' and epoch % 20 == 0 and epoch != self.num_epochs:
                self.model.eval()
                enable_dropout(self.model)
                with torch.no_grad():
                    while batch_gen.has_next():
                        batch_input, _, _, mask = batch_gen.next_batch(batch_size)
                        batch_input, mask = batch_input.to(self.device), mask.to(self.device)
                        mc_probs = []
                        for _ in range(self.forward_times):
                            _, logits = self.model(batch_input, mask)
                            probs = F.softmax(logits[-1].squeeze(0).detach().transpose(2, 1), dim=2)
                            mc_probs.append(probs)
                        mc_probs = torch.stack(mc_probs, dim=0)
                        std = torch.std(mc_probs, dim=0)
                        mc_probs = torch.mean(mc_probs, dim=0)
                        predictions = torch.max(mc_probs, dim=2)[1]
                        # uncertainty_scores = std.gather(2, predictions.unsqueeze(-1))
                        uncertainty_scores = torch.sum(-mc_probs * torch.log(mc_probs), dim=2)
                        batch_gen.update_pseudo_labels(predictions, uncertainty_scores, thres=self.thres)
                        #batch_gen.update_pseudo_labels_all(predictions, uncertainty_scores, thres=self.thres)
                        if self.visualization:
                            data_dir = os.path.join(save_dir, 'train_epoch{}_score'.format(epoch))
                            if not os.path.exists(data_dir):
                                os.makedirs(data_dir)
                            vids, seq_lengths, timestamp_pos = batch_gen.get_useful_info()
                            for b, score in enumerate(uncertainty_scores):
                                np.save(os.path.join(data_dir, '{}.npy'.format(vids[b])), {'timestamp': timestamp_pos[b], 'score': score[:seq_lengths[b]].cpu().numpy()})
                    batch_gen.reset()

                    if self.visualization:
                        self.visualize(batch_gen, save_dir, epoch, batch_size)

            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch) + ".opt")
            writer.add_scalar('trainLoss', epoch_loss / len(batch_gen.list_of_samples), epoch)
            writer.add_scalar('trainAcc', float(correct)/total, epoch)
            print("Training loss = %f,   acc = %f" % (epoch_loss / len(batch_gen.list_of_samples),
                                                               float(correct)/total))

            self.predict(save_dir, epoch, batch_gen.sample_rate)

    def visualize(self, batch_gen, save_dir, epoch, batch_size):
        vis_dir = os.path.join(save_dir, 'train_epoch{}_seg'.format(epoch))
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        while batch_gen.has_next():
            batch_input, batch_target, batch_pseudo_target, mask, _ = batch_gen.next_batch(batch_size)
            batch_input, mask = batch_input.to(self.device), mask.to(self.device)
            _, logits = self.model(batch_input, mask)
            vids, seq_lengths, timestamp_pos = batch_gen.get_useful_info()
            for vi in range(len(batch_input)):
                seq_length = seq_lengths[vi]
                pseudo_target = batch_pseudo_target[vi][:seq_length]
                target = batch_target[vi][:seq_length]
                predicted = torch.max(logits[-1].data, dim=1)[1][vi].squeeze()[:seq_length]
                pseudo_target[pseudo_target==-100] = 8  # 11
                #target[timestamp_pos[vi]] = 9  # 10
                assert len(pseudo_target) == len(target) and len(target) == len(predicted)
                self.segment_bars(os.path.join(vis_dir, '{}.png'.format(vids[vi])), ['gt', target], ['pseudo label', pseudo_target], ['predict', predicted])
        batch_gen.reset()

    def predict(self, model_dir, epoch, sample_rate):
        label2phase_dicts = {
            'cholec80':{
                0: 'Preparation',
                1: 'CalotTriangleDissection',
                2: 'ClippingCutting',
                3: 'GallbladderDissection',
                4: 'GallbladderPackaging',
                5: 'CleaningCoagulation',
                6: 'GallbladderRetraction'},
            'm2cai16':{
                0: 'TrocarPlacement',
                1: 'Preparation',
                2: 'CalotTriangleDissection',
                3: 'ClippingCutting',
                4: 'GallbladderDissection',
                5: 'GallbladderPackaging',
                6: 'CleaningCoagulation',
                7: 'GallbladderRetraction'}
        }
        res_dir = os.path.join(model_dir, 'predict')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            videos = [os.path.join(self.test_features, x) for x in sorted(os.listdir(self.test_features))]
            annotations = [os.path.join(self.test_gt_path, x) for x in sorted(os.listdir(self.test_gt_path))]
            vis_dir = os.path.join(model_dir, 'epoch{}_test_vis'.format(epoch))
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            vid_pre = 41 if  len(videos) == 40 else 28
            label2phase = label2phase_dicts['cholec80' if len(videos)==40 else 'm2cai16']
            #all_pred_phase = []
            #all_label_phase = []
            #correct_phase = 0
            #total_phase = 0

            Acc = AverageMeter()
            Pre = AverageMeter()
            Rec = AverageMeter()
            Jac = AverageMeter()

            for vid, (video, anno) in enumerate(list(zip(videos, annotations))):
                # print(vid)
                features = np.load(video).transpose()
                features = features[:, ::sample_rate]
                with open(anno, 'r') as f:
                    content = f.read().split('\n')
                    if content[-1] == '':
                        content = content[:-1]
                labels = np.zeros(len(content))
                for i in range(len(content)):
                    labels[i] = self.phase2label[content[i].strip().split()[1]]
                labels = torch.Tensor(labels[::sample_rate]).long().to(self.device)

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(self.device)
                _, predictions = self.model(input_x, torch.ones(input_x.size(), device=self.device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()

                # self.segment_bars(os.path.join(vis_dir, '{}.png'.format(video.split('/')[-1].split('.')[0])), ['gt', labels], ['predict', predicted])

                correct_phase = torch.sum(predicted == labels)
                total_phase = len(predicted)

                pred_phase = []
                label_phase = []
                for i in range(len(predicted)):
                    pred_phase.append(int(predicted.data.cpu()[i]))
                for i in range(len(labels)):
                    label_phase.append(int(labels[i]))
                with open(os.path.join(res_dir, f'video{vid+vid_pre}_pred.txt'), 'w') as f:
                    f.write('Frame\tPhase\n')
                    for fid, each_pred in enumerate(predicted):
                        f.write('{}\t{}\n'.format(fid, label2phase[int(each_pred)]))

                accuracy = correct_phase / total_phase
                precision = metrics.precision_score(label_phase, pred_phase, average='macro')
                recall = metrics.recall_score(label_phase, pred_phase, average='macro')
                jaccard = metrics.jaccard_score(label_phase, pred_phase, average='macro')
                #F1 = metrics.f1_score(label_phase, pred_phase, average='macro')
                Acc.update(accuracy)
                Pre.update(precision)
                Rec.update(recall)
                Jac.update(jaccard)

        #accuracy = correct_phase / total_phase
        #precision = metrics.precision_score(all_label_phase, all_pred_phase, average='macro')
        #recall = metrics.recall_score(all_label_phase, all_pred_phase, average='macro')
        #jaccard = metrics.jaccard_score(all_label_phase, all_pred_phase, average='macro')
        #F1 = metrics.f1_score(all_label_phase, all_pred_phase, average='macro')
        print('Evaluating from {} at epoch {}'.format(model_dir, epoch))
        print(tabulate([['{:.2f}'.format(Acc.avg*100), '{:.2f}'.format(Jac.avg*100), '{:.2f}'.format(Pre.avg*100), '{:.2f}'.format(Rec.avg*100)]],
                    headers=['Accuracy', 'Jaccard', 'Precision', 'Recall'], tablefmt='orgtbl'))
        #print('F1 score: {:.4f}'.format(F1))


    def segment_bars_(self, save_path, *labels):
        color_map = [
                (255, 0, 0),    # red
                (255, 165, 0),  # orange
                (255, 255, 0),  # yellow
                (0, 255, 0),    # green
                (0, 127, 255),  # cyan
                (0, 0, 255),    # blue
                (139, 0, 255),  # purple
                (87, 105, 60),  # dark green
                (0, 0, 0),      # black -> background
                (255, 255, 255),# white -> timestamp
                ]
        titles, labels = zip(*labels)
        labels = [label.detach().cpu().numpy() for label in labels]
        nrows = len(labels)
        cmap = plt.cm.Paired
        barprops = dict(aspect='auto', interpolation='none', cmap=cmap)
        figh = 0.15 + 0.15 + (nrows + (nrows-1)*0.15)*0.5
        fig, axs = plt.subplots(nrows=nrows, figsize=(7, figh))
        fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
        for ax, title, label in zip(axs, titles, labels):
            label = np.vstack([label, label])
            #label = (label+0.5) / len(cmap.colors)
            ax.text(-.01, .5, title, va='center', ha='right', fontsize=10, transform=ax.transAxes)
            ax.imshow(label, **barprops)
            ax.set_axis_off()
        fig.savefig(save_path)
        plt.close(fig)

    def segment_bars(self, save_path, *labels):
        def scale_lightness(rgb, scale_l):
            import colorsys
            h, l, s = colorsys.rgb_to_hls(*rgb)
            return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)
        color_map = [
                (255, 0, 0),    # red
                (255, 165, 0),  # orange
                (255, 255, 0),  # yellow
                (0, 255, 0),    # green
                (0, 127, 255),  # cyan
                (0, 0, 255),    # blue
                (139, 0, 255),  # purple
                (87, 105, 60),  # dark green
                (0, 0, 0),      # black -> background
                (255, 255, 255),# white -> timestamp
                ]
        color_map = list(plt.cm.Paired.colors[:8])
        color_map.append((1, 1, 1))
        titles, labels = zip(*labels)
        labels = [label.detach().cpu().numpy().astype(np.int32) for label in labels]
        nrows = len(labels)
        figh = 0.15 + 0.15 + (nrows + (nrows-1)*0.15)*0.5
        fig, axs = plt.subplots(nrows=nrows, figsize=(7, figh))
        fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2, right=0.99)
        for ax, title, label in zip(axs, titles, labels):
            #width = 1 / len(label)
            #x = np.linspace(width/2, 1-width/2, len(label))
            width = 1
            x = np.linspace(0.5, len(label)-0.5, len(label))
            ax.text(-.01, .5, title, va='center', ha='right', fontsize=10, transform=ax.transAxes)
            timestamp = []
            for i, (x_, l_) in enumerate(zip(x, label)):
                if l_ == 9:
                    color = color_map[label[i-1] if i>0 else label[i+1]]
                    timestamp.append((x_, color))
                    # ax.bar([x_], [1], width=width, color=scale_lightness(color, 1.7))
                else:
                    ax.bar([x_], [1], width=width, color=color_map[l_], alpha=0.6)
            for x_, color in timestamp:
                ax.bar([x_], [1], width=width*10, color=scale_lightness(color, 0.8))
            ax.set_axis_off()
        fig.savefig(save_path)
        plt.close(fig)
