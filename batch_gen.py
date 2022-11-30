import os
from copy import deepcopy
import torch
import numpy as np
import random
import torch.nn.functional as F


class BatchGenerator(object):
    def __init__(self, num_classes, phase2label, gt_path, features_path, sample_rate):
        self.last_index = -1
        self.index = 0
        self.num_classes = num_classes
        self.phase2label = phase2label
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}

        self.timestamp = np.load(os.path.join(os.path.dirname(gt_path), "timestamp_middle.npy"), allow_pickle=True).item()
        self.pseudo_labels = dict()
        self.read_data()

        self.max_extend = 0

    def reset(self):
        self.last_index = -1
        self.index = 0
        random.shuffle(self.list_of_samples)

    def has_next(self):
        if self.index < len(self.list_of_samples):
            return True
        return False

    def read_data(self):
        videos = [np.load(os.path.join(self.features_path, x)).transpose() for x in sorted(os.listdir(self.features_path))]
        annotations = [os.path.join(self.gt_path, x) for x in sorted(os.listdir(self.gt_path))]
        self.list_of_samples = list(zip(range(len(videos)), videos, annotations))
        random.shuffle(self.list_of_samples)
        for sample in self.list_of_samples:
            vid, _, anno = sample
            with open(anno, 'r') as f:
                content = f.read().split('\n')
                if content[-1] == '':
                    content = content[:-1]
            labels = np.zeros(len(content), dtype=np.int32)
            for i in range(len(content)):
                labels[i] = self.phase2label[content[i].strip().split()[1]]
            labels = labels[::self.sample_rate]
            self.gt[vid] = labels
            self.pseudo_labels[vid] = torch.ones(labels.shape[0], dtype=torch.long) * (-100)
            for t in self.timestamp[vid]:
                self.pseudo_labels[vid][t] = int(labels[t])

            num_frames = len(labels)

            random_timestamp = self.timestamp[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_timestamp) - 1):
                left_mask[int(labels[random_timestamp[j]]), random_timestamp[j]:random_timestamp[j + 1]] = 1
                right_mask[int(labels[random_timestamp[j + 1]]), random_timestamp[j]:random_timestamp[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])
        # self.generate_confidence_mask()

        print('Num of videos: {}'.format(len(self.list_of_samples)))

    def generate_confidence_mask(self):
            num_frames = len(labels)

            random_timestamp = self.timestamp[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_timestamp) - 1):
                left_mask[int(labels[random_timestamp[j]]), random_timestamp[j]:random_timestamp[j + 1]] = 1
                right_mask[int(labels[random_timestamp[j + 1]]), random_timestamp[j]:random_timestamp[j + 1]] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    def save_pseudo_labels(self, save_dir, it):
        labels = np.save(os.path.join(save_dir, f'pseudo_labels_{it}.npy'), self.pseudo_labels, allow_pickle=True)

    def next_batch(self, batch_size):
        batch = self.list_of_samples[self.index:self.index + batch_size]
        self.last_index = self.index
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        for vid, video, _ in batch:
            # features = np.load(video).transpose()
            features = video
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        batch_pseudo_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            assert np.shape(batch_input[i])[1] == np.shape(batch_target[i])[0]
            seq_length = np.shape(batch_input[i])[1]
            batch_input_tensor[i, :, :seq_length] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :seq_length] = torch.from_numpy(batch_target[i])
            batch_pseudo_tensor[i, :seq_length] = self.pseudo_labels[batch[i][0]]
            mask[i, :, :seq_length] = torch.ones(self.num_classes, seq_length)

        return batch_input_tensor, batch_target_tensor, batch_pseudo_tensor, mask, batch_confidence

    def get_useful_info(self):
        batch = self.list_of_samples[self.last_index:self.index]
        vids = []
        seq_lengths = []
        timestamp_pos = []
        for sample in batch:
            vid, _, _ = sample
            vids.append(vid)
            seq_lengths.append(self.gt[vid].shape[0])
            timestamp_pos.append(self.timestamp[vid])
        return vids,seq_lengths, timestamp_pos

    def get_single_random(self, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_samples[self.last_index: self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_frame = self.timestamp[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def get_average(self, max_frames):
        batch = self.list_of_samples[self.last_index:self.index]
        boundary_target_tensor = torch.ones(len(batch), max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_frame = self.timestamp[vid]
            gt = self.gt[vid]
            gt_tensor = torch.from_numpy(gt.astype(int))
            last_bound = 0
            for i in range(len(single_frame) - 1):
                center = int((single_frame[i] + single_frame[i+1]) / 2)
                boundary_target_tensor[b, last_bound: center] = gt_tensor[single_frame[i]]
                last_bound = center
            boundary_target_tensor[b, last_bound:] = gt_tensor[single_frame[-1]]
        return boundary_target_tensor

    def get_neighbor_label(self, batch_size, max_frames, radius, return_length=False):
        batch = self.list_of_samples[self.index - batch_size:self.index]
        pseudo_label = torch.ones(batch_size, max_frames, dtype=torch.long) * (-100)
        seq_lengths = []
        single_idxs = []
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            single_idxs.append(single_idx)
            vid_gt = self.gt[vid]
            seq_length = vid_gt.shape[0]
            seq_lengths.append(seq_length)
            for i in range(len(single_idx)):
                left_bound = max(0, single_idx[i]-radius)
                if i > 0:
                    left_bound = max(single_idx[i-1]+1, left_bound)
                right_bound = min(seq_length, single_idx[i]+radius)
                if i < len(single_idx)-1:
                    right_bound = min(single_idx[i+1]-1, right_bound)
                pseudo_label[b, left_bound: right_bound+1] = vid_gt[single_idx[i]]
                # print('[{}, {}] printed {}'.format(left_bound, right_bound+1, vid_gt[single_idx[i]]))
        if return_length:
            return pseudo_label, seq_lengths, single_idxs
        return pseudo_label

    def get_boundary_auto(self, batch_size, boundary_score):
        batch = self.list_of_samples[self.index - batch_size:self.index]
        num_video, _, max_frames = boundary_score.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            vid_gt = self.gt[vid]
            score = boundary_score[b].squeeze()
            last_bound = -1
            for i in range(len(single_idx) - 1):
                bound = int(torch.max(score[single_idx[i]:single_idx[i+1]], 0)[1])
                boundary_target_tensor[b, last_bound+1:bound+1] = vid_gt[single_idx[i]]
                last_bound = bound
            boundary_target_tensor[b, last_bound+1:vid_gt.shape[0]] = vid_gt[single_idx[-1]]
        return boundary_target_tensor

    def update_self_labels(self, logits):
        batch = self.list_of_samples[self.last_index: self.index]
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            self.pseudo_labels[vid] = torch.ones(seq_length, dtype=torch.long) * (-100)
            self.pseudo_labels[vid] = torch.max(logits[b], 1)[1][:seq_length]


    def update_pseudo_labels(self, pseudo_labels, uncertainty_scores, thres):
        def maximum_average(seq, min_length, max_length):
            length = len(seq)
            if length <= min_length:
                return 0, length
            max_, s_, e_ = -1, -1, -1
            for i in range(length):
                for j in range(min_length, max_length):
                    if i+j > length:
                        break
                    if seq[i: i+j].mean() > max_:
                        max_ = seq[i: i+j].mean()
                        s_ = i
                        e_ = i+j
            return s_, e_
        self.max_extend += 0.2
        batch = self.list_of_samples[self.last_index: self.index]
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            timestamp = self.timestamp[vid]
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            pseudo_label = pseudo_labels[b].squeeze(0)
            uncertainty_score = uncertainty_scores[b].squeeze(0)
            self.pseudo_labels[vid] = torch.ones(seq_length, dtype=torch.long) * (-100)
            last_right_bound = 0
            for i in range(len(timestamp)):
                left_bound = timestamp[i] - 1
                right_bound = timestamp[i] + 1
                L = 0 if i==0 else timestamp[i-1]+1
                R = seq_length-1 if i==len(timestamp)-1 else timestamp[i+1]-1
                while left_bound >= L and ((pseudo_label[left_bound] == video_gt[timestamp[i]] and uncertainty_score[left_bound] < thres)):# \
                        #or (pseudo_label[left_bound] != video_gt[timestamp[i]] and uncertainty_score[left_bound] > 1)): # \
                            #and torch.sum(pseudo_label[max(L, left_bound-15):left_bound]==video_gt[timestamp[i]])/(left_bound-max(L, left_bound-15))>0.6)):
                    left_bound -= 1
                while right_bound <=R and ((pseudo_label[right_bound] == video_gt[timestamp[i]] and uncertainty_score[right_bound] < thres)):# \
                        #or (pseudo_label[right_bound] != video_gt[timestamp[i]] and uncertainty_score[right_bound] > 1)): \
                            #and torch.sum(pseudo_label[right_bound:min(R, right_bound+15)]==video_gt[timestamp[i]])/(min(R, right_bound+15)-right_bound)>0.6)):
                    right_bound += 1
                # left_bound = max(left_bound, timestamp[i] - int(self.max_extend * (timestamp[i]-L)))
                # right_bound = min(right_bound, timestamp[i] + int(self.max_extend * (R-timestamp[i])))
                self.pseudo_labels[vid][left_bound+1: right_bound] = int(video_gt[timestamp[i]])

                #if i > 0:
                #    min_length = max(4, int(0.02 * (timestamp[i] - timestamp[i-1])))
                #    max_length = max(10, int(0.05 * (timestamp[i] - timestamp[i-1])))
                #    l, r = maximum_average(uncertainty_score[timestamp[i-1]+1: timestamp[i]], min_length, max_length)
                #    l += timestamp[i-1] + 1
                #    r += timestamp[i-1] + 1
                #    self.pseudo_labels[vid][l: r] = -100
                    #self.pseudo_labels[vid][last_right_bound: l] = int(video_gt[timestamp[i]])
                    #self.pseudo_labels[vid][r: left_bound+1] = int(video_gt[timestamp[i]])
                last_right_bound = right_bound

    def update_pseudo_labels_all(self, pseudo_labels, uncertainty_scores, thres):
        batch = self.list_of_samples[self.last_index: self.index]
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            pseudo_label = pseudo_labels[b].squeeze(0)
            uncertainty_score = uncertainty_scores[b].squeeze(0)
            self.pseudo_labels[vid] = torch.ones(seq_length, dtype=torch.long) * (-100)
            for j in range(seq_length):
                if uncertainty_score[j] < thres:
                    self.pseudo_labels[vid][j] = pseudo_label[j]

    def update_pseudo_labels_noisy(self):
        noisy_length = 10
        for sample in self.list_of_samples:
            vid, _, _ = sample
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            timestamp = self.timestamp[vid]
            self.pseudo_labels[vid] = torch.ones(seq_length, dtype=torch.long) * (-100)
            for i in range(len(timestamp)):
                pos = timestamp[i]
                L = 0 if i==0 else timestamp[i-1]+1
                R = seq_length-1 if i==len(timestamp)-1 else timestamp[i+1]-1
                left = min(max(0, pos-L-noisy_length), noisy_length)
                right = min(max(R-pos-noisy_length, 1), noisy_length)
                choice = np.concatenate([
                    np.random.choice(list(range(pos-left, pos)), left//2, replace=False),
                    np.array([pos]),
                    np.random.choice(list(range(pos+1, pos+right)), right//2, replace=False)
                ])
                self.pseudo_labels[vid][pos-left: pos+right] = video_gt[pos]


    def update_pseudo_labels_truth(self, rate):
        for sample in self.list_of_samples:
            vid, _, _ = sample
            video_gt = self.gt[vid]
            seq_length = video_gt.shape[0]
            timestamp = self.timestamp[vid]
            self.pseudo_labels[vid] = torch.ones(seq_length, dtype=torch.long) * (-100)
            for i in range(len(timestamp)):
                pos = timestamp[i]
                L = pos - 1
                while L >= 0 and video_gt[L] == video_gt[pos]:
                    L -= 1
                R = pos + 1
                while R < seq_length and video_gt[R] == video_gt[pos]:
                    R += 1
                left = int(rate * (pos-L-1))
                right = int(rate * (R-pos))
                self.pseudo_labels[vid][pos-left: pos+right] = video_gt[pos]


    def get_extend_label(self, batch_prob, thres, return_length=False):
        batch_size, max_frames, _ = batch_prob.size()
        batch = self.list_of_samples[self.last_index: self.index]
        pseudo_label = torch.ones(batch_size, max_frames, dtype=torch.long) * (-100)
        seq_lengths = []
        single_idxs = []
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            single_idxs.append(single_idx)
            vid_gt = self.gt[vid]
            seq_length = vid_gt.shape[0]
            seq_lengths.append(seq_length)
            #pred = batch_pred[b]
            #prob = F.softmax(pred.transpose(1, 0), dim=1)
            prob = batch_prob[b]
            entropy = torch.sum(- prob * torch.log(prob), dim=1)
            predicted = torch.max(prob, dim=1)[1]
            for i in range(len(single_idx)):
                left_bound = single_idx[i] - 1
                right_bound = single_idx[i] + 1
                L = 0 if i==0 else single_idx[i-1]+1
                R = seq_length-1 if i==len(single_idx)-1 else single_idx[i+1]-1
                while left_bound >= L and \
                        predicted[left_bound] == vid_gt[single_idx[i]] and entropy[left_bound] < thres:
                    left_bound -= 1
                while right_bound <= R and \
                        predicted[right_bound] == vid_gt[single_idx[i]] and entropy[right_bound] < thres:
                    right_bound += 1
                pseudo_label[b, left_bound+1: right_bound] = vid_gt[single_idx[i]]
                # print('[{}, {}] printed {}'.format(left_bound, right_bound+1, vid_gt[single_idx[i]]))
        if return_length:
            return pseudo_label, seq_lengths, single_idxs
        return pseudo_label

    def get_uncertainty_label(self, batch_pred, min_length_rate, max_length_rate, return_length=False):
        batch_size, _, max_frames = batch_pred.size()
        batch = self.list_of_samples[self.index - batch_size:self.index]
        pseudo_label = torch.ones(batch_size, max_frames, dtype=torch.long) * (-100)
        seq_lengths = []
        single_idxs = []
        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            single_idxs.append(single_idx)
            vid_gt = self.gt[vid]
            seq_length = vid_gt.shape[0]
            seq_lengths.append(seq_length)
            last_bound = -1
            pred = batch_pred[b]
            prob = F.softmax(pred.transpose(1, 0), dim=1)
            entropy = torch.sum(- prob * torch.log(prob), dim=1)
            prefix = entropy
            for i in range(1, seq_length):
                prefix[i] += prefix[i-1]
            for i in range(len(single_idx)-1):
                min_length = int((single_idx[i+1] - single_idx[i]) * min_length_rate)
                max_length = int((single_idx[i+1] - single_idx[i]) * max_length_rate)
                if single_idx[i+1] - single_idx[i] <= min_length:
                    pseudo_label[b, last_bound+1:single_idx[i]+1] = vid_gt[single_idx[i]]
                    last_bound = single_idx[i+1] - 1
                else:
                    max_u, s_, e_ = -1e9, -1, -1
                    for s in range(single_idx[i]+1, single_idx[i+1]):
                        for e in range(s+min_length-1, s+max_length-1):
                            if e >= single_idx[i+1]:
                                break
                            avg = (prefix[e]-prefix[s-1]) / (e-s+1)
                            if avg > max_u:
                                max_u, s_, e_ = avg, s, e
                    pseudo_label[b, last_bound+1:s_] = vid_gt[single_idx[i]]
                    last_bound = e_
            pseudo_label[b, last_bound+1:seq_length] = vid_gt[single_idx[-1]]
        if return_length:
            return deepcopy(pseudo_label), deepcopy(seq_lengths), deepcopy(single_idxs)
        return deepcopy(pseudo_label)

    def get_boundary(self, batch_size, pred):
        # This function is to generate pseudo labels

        batch = self.list_of_samples[self.last_index:self.index]
        num_video, _, max_frames = pred.size()
        boundary_target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)

        for b, sample in enumerate(batch):
            vid, _, _ = sample
            single_idx = self.timestamp[vid]
            vid_gt = self.gt[vid]
            features = pred[b]
            boundary_target = np.ones(vid_gt.shape) * (-100)
            boundary_target[:single_idx[0]] = vid_gt[single_idx[0]]  # frames before first single frame has same label
            left_bound = [0]

            # Forward to find action boundaries
            for i in range(len(single_idx) - 1):
                start = single_idx[i]
                end = single_idx[i + 1] + 1
                left_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(start + 1, end):
                    center_left = torch.mean(features[:, left_bound[-1]:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:end], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    left_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(left_score) + start + 1
                left_bound.append(cur_bound.item())

            # Backward to find action boundaries
            right_bound = [vid_gt.shape[0]]
            for i in range(len(single_idx) - 1, 0, -1):
                start = single_idx[i - 1]
                end = single_idx[i] + 1
                right_score = torch.zeros(end - start - 1, dtype=torch.float)
                for t in range(end - 1, start, -1):
                    center_left = torch.mean(features[:, start:t], dim=1)
                    diff_left = features[:, start:t] - center_left.reshape(-1, 1)
                    score_left = torch.mean(torch.norm(diff_left, dim=0))

                    center_right = torch.mean(features[:, t:right_bound[-1]], dim=1)
                    diff_right = features[:, t:end] - center_right.reshape(-1, 1)
                    score_right = torch.mean(torch.norm(diff_right, dim=0))

                    right_score[t-start-1] = ((t-start) * score_left + (end - t) * score_right)/(end - start)

                cur_bound = torch.argmin(right_score) + start + 1
                right_bound.append(cur_bound.item())

            # Average two action boundaries for same segment and generate pseudo labels
            left_bound = left_bound[1:]
            right_bound = right_bound[1:]
            num_bound = len(left_bound)
            for i in range(num_bound):
                temp_left = left_bound[i]
                temp_right = right_bound[num_bound - i - 1]
                middle_bound = int((temp_left + temp_right)/2)
                boundary_target[single_idx[i]:middle_bound] = vid_gt[single_idx[i]]
                boundary_target[middle_bound:single_idx[i + 1] + 1] = vid_gt[single_idx[i + 1]]

            boundary_target[single_idx[-1]:] = vid_gt[single_idx[-1]]  # frames after last single frame has same label
            boundary_target_tensor[b, :vid_gt.shape[0]] = torch.from_numpy(boundary_target)

        return boundary_target_tensor
