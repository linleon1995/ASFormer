'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
from grid_sampler import GridSampler, TimeWarpLayer
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler


class VAS_Dataset(Dataset):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, list_of_samples):
        self.list_of_samples = list_of_samples
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.list_of_samples)

    def __getitem__(self, idx):
        vid = self.list_of_samples[idx]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        input = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]
        seq_length = input.shape[1]
        mask = np.ones((self.num_classes, seq_length), np.float32)
        return {'input': input, 'target': target, 'mask': mask, 'seq_length': seq_length, 'vid': vid}


def align_video_length(batch):
    seq_length_list = [sample['seq_length'] for sample in batch]
    vid_list = [sample['vid'] for sample in batch]
    N = len(batch)  # batch_size
    C = batch[0]['input'].shape[0]  # feature size
    L = max(seq_length_list)  # sequence length
    K = batch[0]['mask'].shape[0]  # number of class

    batch_input = torch.zeros([N, C, L], dtype=torch.float)
    batch_target = -100 * torch.ones([N, L], dtype=torch.long)
    batch_mask = torch.zeros([N, K, L], dtype=torch.float)
    for idx, sample in enumerate(batch):
        sample_length = seq_length_list[idx]
        batch_input[idx, :, :sample_length] = torch.from_numpy(
            np.array(sample['input'], np.float32))
        batch_target[idx, :sample_length] = torch.from_numpy(
            np.array(sample['target'], np.int64))
        batch_mask[idx, :, :sample_length] = torch.from_numpy(
            np.array(sample['mask'], np.float32))

    return batch_input, batch_target, batch_mask, vid_list


def get_train_dataloader(
        files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, batch_size, num_workers, seed):

    np.random.seed(seed)
    train_size = int(1.0*len(files))
    train_files = np.random.choice(files, size=train_size)
    valid_files = list(set(files)-set(train_files))

    train_loader = get_vas_dataloader(
        train_files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, batch_size, num_workers
    )
    valid_loader = get_vas_dataloader(
        valid_files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, batch_size, num_workers
    )
    return train_loader, valid_loader


def get_vas_dataloader(
        files, num_classes, actions_dict, gt_path, features_path,
        sample_rate, batch_size, num_workers):

    samples = files
    # samples = read_data(files)
    dataset = VAS_Dataset(num_classes, actions_dict, gt_path,
                          features_path, sample_rate, samples)
    loader = DataLoader(
        dataset=dataset, collate_fn=align_video_length, batch_size=batch_size, num_workers=num_workers)
    return loader


def read_data(vid_list_file):
    file_ptr = open(vid_list_file, 'r')
    list_of_samples = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    return list_of_samples


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        self.timewarp_layer = TimeWarpLayer()

    def reset(self):
        self.index = 0
        self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        self.gts = [self.gt_path + vid for vid in self.list_of_examples]
        self.features = [self.features_path +
                         vid.split('.')[0] + '.npy' for vid in self.list_of_examples]
        self.my_shuffle()

    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)

    def warp_video(self, batch_input_tensor, batch_target_tensor):
        '''
        :param batch_input_tensor: (bs, C_in, L_in)
        :param batch_target_tensor: (bs, L_in)
        :return: warped input and target
        '''
        bs, _, T = batch_input_tensor.shape
        grid_sampler = GridSampler(T)
        grid = grid_sampler.sample(bs)
        grid = torch.from_numpy(grid).float()

        warped_batch_input_tensor = self.timewarp_layer(
            batch_input_tensor, grid, mode='bilinear')
        batch_target_tensor = batch_target_tensor.unsqueeze(1).float()
        warped_batch_target_tensor = self.timewarp_layer(
            batch_target_tensor, grid, mode='nearest')  # no bilinear for label!
        warped_batch_target_tensor = warped_batch_target_tensor.squeeze(
            1).long()  # obtain the same shape

        return warped_batch_input_tensor, warped_batch_target_tensor

    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))

    # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
    def next_batch(self, batch_size, if_warp=False):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]

        self.index += batch_size

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            file_ptr = open(batch_gts[idx], 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]

            feature = features[:, ::self.sample_rate]
            target = classes[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[
                                         0], max(length_of_sequences), dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(
            length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(
            length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(
                    batch_input[i]).unsqueeze(0), torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i, :np.shape(
                    batch_target[i])[0]] = warped_input.squeeze(0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[
                    1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[
                    0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask, batch


if __name__ == '__main__':
    pass
