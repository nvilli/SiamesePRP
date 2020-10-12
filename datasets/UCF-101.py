# 2020.10.07 UCF-101 dataset
# 2020.10.09 update __getitem__, not need to reconstruct
# 2020.10.11 update __getitem__, just two sample step choices

import os
import torch.utils.data as data
import cv2
import sys
import random
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse
import collections
# from datasets import patch_region
envs = os.environ
sys.path.append('..')


class ucf101(data.Dataset):
    def __init__(self, root, mode='train', args=None):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop(112),
            transforms.ToTensor()
        ])

        self.root = root
        self.mode = mode
        self.args = args
        self.toPIL = transforms.ToPILImage()
        self.tensortrans = transforms.Compose([transforms.ToTensor()])

        self.split = '1'

        train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
        self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
        self.test_split = pd.read_csv(test_split_path, header=None)[0]

        if mode == 'train':
            self.list = self.train_split
        else:
            self.list = self.test_split

        self.batch_size = 8

        # sample step
        # self.sample_step_list = [1, 2, 4, 8]
        self.sample_step_list = [1, 2]

        self.MA_mode = 'DPAU'

    def __getitem__(self, index):
        
        
        # if len(self.)
        seed = random.randint(0, 9)

        # sample_clip: clip that sampled with different sample clip
        # sample_step_label: sample step as a label

        '''
        # no need to be samples
        if (seed <= 4):
            videodata, sample_step_label = self.loadcvvideo_Finsert(index, sample_step=None)
            print('seed <= 4, sample step label', sample_step_label)
            target_clip = self.crop(videodata)
            sample_step = self.sample_step_list[sample_step_label]
            sample_inds = torch.arange(0, len(videodata), step=sample_step)
            sample_clip = target_clip[:, sample_inds, :, :]

        if (seed > 4):
            videodata, sample_step_label = self.loadcvvideo_Finsert(index, sample_step=1)
            print('seed > 4, sample step label', sample_step_label)
            target_clip = self.crop(videodata)
            sample_step = self.sample_step_list[sample_step_label]
            sample_inds = torch.arange(0, len(videodata), step=sample_step)
            sample_clip = target_clip[:, sample_inds, :, :]
        '''

        videodata, sample_step_label = self.loadcvvideo_Finsert(index, sample_step=None)
        target_clip = self.crop(videodata)
        sample_step = self.sample_step_list[sample_step_label]
        sample_inds = torch.arange(0, len(videodata), step=sample_step)
        sample_clip = target_clip[:, sample_inds, :, :]
        print('sample_step_label: ', sample_step_label, 'sample step: ', self.sample_step_list[sample_step_label], 'sample length: ', len(videodata))


        return sample_clip, sample_step_label

    def loadcvvideo_Finsert(self, index, sample_step=None):
        need = 16
        fname = self.list[index]
        fname = os.path.join(self.root, 'video', fname)
        # print(fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(frame_count)


        if sample_step is None:
            # print('sample step is none')
            sample_step_label = np.random.randint(low=0, high=len(self.sample_step_list))
            sample_step = self.sample_step_list[sample_step_label]
            # print(sample_step)
        else:
            sample_step_label = self.sample_step_list.index(sample_step)
        
        sample_len = need * sample_step
        shortest_len = sample_len + 1

        # if video frame count is less than the number of video that we sample
        while frame_count < shortest_len:
            # print("frame count is less than shortest len", frame_count, ' ', shortest_len)
            index = np.random.randint(self.__len__())
            fname = self.list[index]
            fname = os.path.join(self.root, 'video', fname)
            capture = cv2.VideoCapture(fname)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        start = np.random.randint(0, frame_count - shortest_len + 1)

        if start > 0:
            start = start - 1
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while(sample_count < sample_len and retaining):
            retaining, frame = capture.read()
            if retaining is False:
                count += 1
                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1
        capture.release()

        while len(buffer) < sample_len:
            index = np.random.randint(self.__len__())
            print('retaining:{} buffer_len:{} sample_len{}'.format(retaining, len(buffer), sample_len))
            buffer, sample_step_label = self.loadcvvideo_Finsert(index, sample_step)
            print('reload')

        return buffer, sample_step_label

    def crop(self, frames):
        video_clips = []
        seed = random.random()
        for frame in frames:
            random.seed(seed)
            frame = self.toPIL(frame)
            frame = self.transforms(frame)
            video_clips.append(frame)
        clip = torch.stack(video_clips).permute(1, 0, 2, 3)

        return clip

    def __len__(self):
        return len(self.list)


def parse_args():
    parser = argparse.ArgumentParser(description='video clip')
    parser.add_argument('--lpls',     type=bool, default=False,     help='use lpls_loss or not')
    parser.add_argument('--msr',      type=bool, default=False,     help='use multi sample rate or not')
    parser.add_argument('--vcop',     type=bool, default=True,      help='predict video clip order or not')
    parser.add_argument('--gpu',      type=str,  default='0',       help='gpu id')
    parser.add_argument('--epoch',    type=int,  default=300,       help='number of total epochs to run')
    parser.add_argument('--exp_name', type=str,  default='default', help='experiment name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    com = ucf101('/home/guojie/Dataset/UCF-101-origin', mode='train', args=args)
    print(com.root)
    train_dataloader = DataLoader(com, batch_size=8, num_workers=1, shuffle=True, drop_last=True)


    for i, (clip1, sample_step_label) in enumerate(train_dataloader):
        print('i: ', i)
        '''
        print('clip size: ', clip1.size())
        print('sample_step_label: ', sample_step_label)
        '''