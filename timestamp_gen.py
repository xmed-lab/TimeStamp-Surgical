import os
import argparse
import random
import numpy as np


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
sample_rate = 1


def timestamp_gen(dataset, anno_path, save_path, type_):
    #random.seed(20000604)
    phase2label = phase2label_dicts[dataset]
    anno_files = [os.path.join(anno_path, x) for x in sorted(os.listdir(anno_path))]
    vids = range(len(anno_files))
    vid2timestamp = {}
    for idx, anno_file in enumerate(anno_files):
        with open(anno_file, 'r') as f:
            content = f.read().split('\n')
            if content[-1] == '':
                content = content[:-1]
        labels = [phase2label[line.strip().split()[1]] for line in content]
        labels = labels[::sample_rate]
        last = 0
        cur = 1
        timestamp = list()
        while cur < len(labels):
            if labels[cur] != labels[cur-1]:
                timestamp.append(random.randint(last, cur-1))
                last = cur
            cur += 1
        if type_ == 'random':
            timestamp.append(random.randint(last, cur-1))
        elif type_ == 'start':
            timestamp.append(last)
        elif type_ == 'end':
            timestamp.append(cur-1)
        elif type_ == 'middle':
            timestamp.append((last+cur-1)//2)
        else:
            raise("Invalid timestamp type: {}!".format(type_))
        vid2timestamp[vids[idx]] = timestamp
    print('timestamp annotations are save to {}'.format(save_path))
    np.save(save_path, vid2timestamp, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("TimeStamp generator")
    parser.add_argument('--action', type=str, help='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--anno', type=str, help='path to full annotations')
    parser.add_argument('--save', type=str, help='path to save timestamp annotations')
    args = parser.parse_args()
    timestamp_gen(args.dataset, args.anno, args.save, args.action)
