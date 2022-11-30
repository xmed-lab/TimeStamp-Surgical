import os
import random
import numbers
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import torchvision.transforms.functional as TF

phase2label_dicts = {
    'cholec80':{
    'Preparation':0,
    'CalotTriangleDissection':1,
    'ClippingCutting':2,
    'GallbladderDissection':3,
    'GallbladderPackaging':4,
    'CleaningCoagulation':5,
    'GallbladderRetraction':6},
    
    'm2cai16':{
    'TrocarPlacement':0,
    'Preparation':1,
    'CalotTriangleDissection':2,
    'ClippingCutting':3,
    'GallbladderDissection':4,
    'GallbladderPackaging':5,
    'CleaningCoagulation':6,
    'GallbladderRetraction':7}
}
train_split = {
    'cholec80': 40,
    'm2cai16': 27,
}
stats_dict = {
    'cholec80':{
        'mean': [0.41757566,0.26098573,0.25888634],
        'std': [0.21938758,0.1983,0.19342837]
        },
    'm2cai16':{
        'mean': [0.4025188, 0.2787089, 0.26787987],
        'std': [0.2016145, 0.16719624, 0.16052364]
    }
}


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0
    
    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
        w, h = img.size
        th, tw = self.size
        if w == tw and h ==th:
            return img
        random.seed(self.count)
        x1 = random.randint(0, w-tw)
        y1 = random.randint(0, h-th)
        self.count += 1
        return img.crop((x1, y1, x1+tw, y1+th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        random.seed(self.count)
        prob = random.random()
        self.count += 1
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        random.seed(self.count)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)


class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        random.seed(self.count)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)
        return img_


def base_transform(dataset):
    mean, std = stats_dict[dataset]['mean'], stats_dict[dataset]['std']
    return transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(),
            RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


class PureTimestampDataset(Dataset):
    def __init__(self, dataset, root, anno_dir, timestamp, transform=None):
        self.dataset = dataset
        self.imgs = []
        self.labels = []
        videos = [os.path.join(root, x) for x in sorted(os.listdir(root))]
        videos = videos[: train_split[dataset]]  #  trainset
        annos = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir)) if x.startswith('video') or x.startswith('workflow')]
        video2timestamp = np.load(timestamp, allow_pickle=True).item()
        for i in range(len(videos)):
            timestamp = video2timestamp[i]
            frames = [os.path.join(videos[i], x) for x in sorted(os.listdir(videos[i]))]
            with open(annos[i], 'r') as f:
                content = f.read().split('\n')[:-1]
            labels = [phase2label_dicts[self.dataset][x.split('\t')[1]] for x in content]
            for idx, fid in enumerate(timestamp):
                self.imgs.append(frames[fid])
                self.labels.append(labels[fid])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_transform()
    def __getitem__(self, index):
        img = self.transform(default_loader(self.imgs[index]))
        label = self.labels[index]
        return img, label, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        mean = stats_dict[self.dataset]['mean']
        std = stats_dict[self.dataset]['std']
        return transforms.Compose([
                transforms.Resize((250, 250)),
                RandomCrop(224),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomHorizontalFlip(),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                #transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
            ])

class PseudoLabelDataset(Dataset):
    def __init__(self, dataset, root, anno_dir, pseudo_label, transform=None, sample_rate=1):
        self.dataset = dataset
        self.imgs = []
        self.labels = []
        videos = [os.path.join(root, x) for x in sorted(os.listdir(root))]
        videos = videos[: train_split[dataset]]  #  trainset
        annos = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir)) if x.startswith('video') or x.startswith('workflow')]
        video2pseudo = np.load(pseudo_label, allow_pickle=True).item()
        for i in range(len(videos)):
            pseudo_label = video2pseudo[i]
            frames = [os.path.join(videos[i], x) for x in sorted(os.listdir(videos[i]))]
            for fid, p_l in enumerate(pseudo_label):
                if fid % sample_rate == 0:
                    if p_l != -100:
                        self.imgs.append(frames[fid])
                        self.labels.append(p_l)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_transform()
    def __getitem__(self, index):
        img = self.transform(default_loader(self.imgs[index]))
        label = self.labels[index]
        return img, label, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def get_transform(self):
        mean = stats_dict[self.dataset]['mean']
        std = stats_dict[self.dataset]['std']
        return transforms.Compose([
                transforms.Resize((250, 250)),
                RandomCrop(224),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomHorizontalFlip(),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
                #transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
            ])

class FullDataset(Dataset):
    def __init__(self, dataset, root, anno_dir, train, unsupervised=False, timestamp=None, transform=None, sample_rate=1):
        self.dataset = dataset
        self.imgs = []
        self.labels = []
        videos = [os.path.join(root, x) for x in sorted(os.listdir(root))]
        annos = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir)) if x.startswith('video') or x.startswith('workflow')]
        if train:
            videos, annos = videos[: train_split[dataset]], annos[: train_split[dataset]]
        else:
            videos, annos = videos[train_split[dataset]: ], annos[train_split[dataset]: ]
        if unsupervised:
            video2timestamp = np.load(timestamp, allow_pickle=True).item()

        for i in range(len(videos)):
            frames = [os.path.join(videos[i], x) for x in sorted(os.listdir(videos[i]))]
            with open(annos[i], 'r') as f:
                content = f.read().split('\n')[:-1]
            labels = [phase2label_dicts[self.dataset][x.split('\t')[1]] for x in content]
            if self.dataset == 'cholec80':
                frames = frames[:-1]
            assert len(labels) == len(frames), \
                    '{} ambiguou length: labels {} while frames {}'.format(videos[i], len(labels), len(frames))
            if unsupervised:
                tmp_frames = []
                tmp_labels = []
                for j in range(len(frames)):
                    if j not in video2timestamp[i]:
                        tmp_frames.append(frames[j])
                        tmp_labels.append(labels[j])
                frames = tmp_frames
                labels = tmp_labels
            frames = frames[::sample_rate]
            labels = labels[::sample_rate]
            self.imgs.extend(frames)
            self.labels.extend(labels)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.get_transform(train)

    def __getitem__(self, index):
        img = default_loader(self.imgs[index])
        if self.transform is not None:
            return self.transform(img), self.labels[index], self.imgs[index]
        return img, self.labels[index], self.imgs[index]

    def __len__(self):
        return len(self.imgs)

    def get_transform(self, train):
        mean = stats_dict[self.dataset]['mean']
        std = stats_dict[self.dataset]['std']
        if train:
            return transforms.Compose([
                    transforms.Resize((250, 250)),
                    RandomCrop(224),
                    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                    RandomHorizontalFlip(),
                    RandomRotation(5),
                    transforms.ToTensor(),
                    #transforms.Normalize([0.41757566,0.26098573,0.25888634],[0.21938758,0.1983,0.19342837])
                    transforms.Normalize(mean, std)
                ])
        else:
            return transforms.Compose([
                    transforms.Resize((250, 250)),
                    RandomCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])


class BatchGenerator(object):
    def __init__(self, dataset, root, anno_dir, timestamp, batch_size, sample_rate=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.unsup_batch_size = batch_size
        self.seg_batch_size = 2
        self.seg_length = 8
        self.timestamp = np.load(timestamp, allow_pickle=True).item()
        videos = [os.path.join(root, x) for x in sorted(os.listdir(root))]
        videos = videos[: train_split[dataset]]
        annos = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir))]
        annos = annos[: train_split[dataset]]
        self.num_videos = len(videos)
        self.video_lengths = []
        self.video_labels = []
        self.timestamp_imgs = []
        self.timestamp_labels = []
        self.unsup_imgs = []
        self.seg_imgs = []
        self.total_imgs = []
        self.total_imgs_ = []
        self.total_labels_ = []
        self.imgs = []
        self.labels = []
        for i in range(self.num_videos):
            frames = [os.path.join(videos[i], x) for x in sorted(os.listdir(videos[i]))]
            with open(annos[i], 'r') as f:
                content = f.read().split('\n')[:-1]
            labels = [phase2label_dicts[self.dataset][x.split('\t')[1]] for x in content]
            if self.dataset == 'cholec80':
                frames = frames[:-1]
            self.total_imgs.extend(frames)
            self.total_imgs_.append(frames)
            self.total_labels_.append(labels)
            self.video_lengths.append(len(frames))
            for fid in self.timestamp[i]:
                self.timestamp_imgs.append(frames[fid])
                self.timestamp_labels.append(labels[fid])
                self.imgs.append(frames[fid])
                self.labels.append(labels[fid])
            for fid, frame in enumerate(frames):
                if fid not in self.timestamp[i]:
                    self.unsup_imgs.append(frame)
            self.seg_imgs.extend([frames[i: i+self.seg_length] for i in range(0, len(frames), 2*self.seg_length)])
            if len(self.seg_imgs[-1]) != self.seg_length:
                self.seg_imgs.pop()
        self.unsup_imgs = self.unsup_imgs[::sample_rate]
        self.transform = self.get_transform()

        self.reset()
        self.reset_unsup()
        self.reset_total()
        self.reset_seg()

        print('Num of labeled imgs: {}\nNum of unlabeled imgs: {}\nNum of segment imgs: {}'.format(len(self.imgs), len(self.unsup_imgs), len(self.seg_imgs)))

    def reset_total(self):
        self.total_index = 0

    def reset_unsup(self):
        self.unsup_index = 0
        random.shuffle(self.unsup_imgs)

    def reset_seg(self):
        self.seg_index = 0
        random.shuffle(self.seg_imgs)

    def has_next_seg(self):
        return self.seg_index < len(self.seg_imgs)

    def has_next_unsup(self):
        return self.unsup_index < len(self.unsup_imgs)

    def has_next_total(self):
        return self.total_index < len(self.total_imgs)

    def next_batch_total(self):
        total_next_index = min(self.total_index+self.batch_size, len(self.total_imgs))
        data = torch.stack([self.transform(default_loader(self.total_imgs[i])) for i in range(self.total_index, total_next_index)])
        self.total_index = total_next_index
        return data
    
    def next_batch_unsup(self):
        unsup_next_index = min(self.unsup_index+self.unsup_batch_size, len(self.unsup_imgs))
        data = torch.stack([self.transform(default_loader(self.unsup_imgs[i])) for i in range(self.unsup_index, unsup_next_index)])
        self.unsup_index = unsup_next_index
        return data

    def next_batch_seg(self):
        seg_next_index = min(self.seg_index+self.seg_batch_size, len(self.seg_imgs))
        data = torch.cat([torch.stack([self.transform(default_loader(x)) for x in self.seg_imgs[i]]) for i in range(self.seg_index, seg_next_index)])
        self.seg_index = seg_next_index
        return data

    def update_dataset(self, predictions, scores, thres):
        self.imgs = []
        self.labels = []
        self.unsup_imgs = []
        labels = []
        for i in range(self.num_videos):
            timestamp = self.timestamp[i]
            prediction = predictions[i]
            score = scores[i]
            last_right_bound = 0
            for j in range(len(timestamp)):
                left_bound = timestamp[j] - 1
                right_bound = timestamp[j] + 1
                L = 0 if j==0 else timestamp[j-1]+1
                R = self.video_lengths[i]-1 if j==len(timestamp)-1 else timestamp[j+1]-1
                this_label = self.total_labels_[i][timestamp[j]]
                while left_bound >= L and prediction[left_bound] == this_label and score[left_bound] < thres:
                    left_bound -= 1
                while right_bound <= R and prediction[right_bound] == this_label and score[right_bound] < thres:
                    right_bound += 1
                self.imgs.extend(self.total_imgs_[i][left_bound+1: right_bound])
                self.labels.extend([this_label for _ in range(right_bound-left_bound-1)])
                self.unsup_imgs.extend(self.total_imgs_[i][last_right_bound: left_bound+1])
                labels.extend(self.total_labels_[i][left_bound+1: right_bound])
                last_right_bound = right_bound
            self.unsup_imgs.extend(self.total_imgs[i][right_bound:])
        print('pseudo labels\' accuracy: {}'.format((np.array(self.labels)==np.array(labels)).sum() / len(labels)))
        assert len(self.imgs) == len(self.labels)

    def reset(self):
        self.index = 0
        indicies = list(np.arange(len(self.imgs)))
        random.shuffle(indicies)
        self.imgs = np.array(self.imgs)[indicies]
        self.labels = np.array(self.labels)[indicies]

    def has_next(self):
        return self.index < len(self.imgs)

    def next_batch(self):
        next_index = min(self.index+self.batch_size, len(self.imgs))
        data = torch.stack([self.transform(default_loader(self.imgs[i])) for i in range(self.index, next_index)])
        target = torch.tensor([self.labels[i] for i in range(self.index, next_index)]).long()
        self.index = next_index
        return data, target

    def get_transform(self):
        mean = stats_dict[self.dataset]['mean']
        std = stats_dict[self.dataset]['std']
        return transforms.Compose([
                transforms.Resize((250, 250)),
                RandomCrop(224),
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                RandomHorizontalFlip(),
                RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])



if __name__ == '__main__':
    # unit-test
    dataset = PureTimestampDataset(
            dataset='cholec80',
            root='./dataset/cholec80/frames',
            timestamp='./dataset/cholec80/frames_annotations/timestamp.npy'
    )
    img, label, img_path = dataset[0]
    print('Timestamp Dataset:\nTotal: {}\nExample: {}, {}, {}\n'.format(len(dataset), img.size(), label, img_path))

    dataset = FullDataset(
            dataset='cholec80',
            root='./dataset/cholec80/frames',
            anno_dir='./dataset/cholec80/frames_annotations',
            train=True
    )
    img, label, img_path = dataset[0]
    print('Full Train Dataset:\nTotal: {}\nExample: {}, {}, {}\n'.format(len(dataset), img.size(), label, img_path))

 
    dataset = FullDataset(
            dataset='cholec80',
            root='./dataset/cholec80/frames',
            anno_dir='./dataset/cholec80/frames_annotations',
            train=False
    )
    img, label, img_path = dataset[0]
    print('Full Test Dataset:\nTotal: {}\nExample: {}, {}, {}\n'.format(len(dataset), img.size(), label, img_path))
