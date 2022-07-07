# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import random
from abc import ABC
from math import ceil

from loguru import logger
from tqdm import tqdm
import numpy as np


class BaseDataLoader(ABC):
    """Abstract class of dataloader

    Notes:
        ``'scale'`` can be set in config to limit the size of dataset.

    """

    def __init__(self, opt, dataset):
        """
        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: dataset

        """
        self.opt = opt
        self.dataset = dataset
        self.scale = opt.get('scale', 1)
        assert 0 < self.scale <= 1

    def get_data(self, batch_fn, batch_size, shuffle=True, process_fn=None):
        """Collate batch data for system to fit

        Args:
            batch_fn (func): function to collate data
            batch_size (int):
            shuffle (bool, optional): Defaults to True.
            process_fn (func, optional): function to process dataset before batchify. Defaults to None.

        Yields:
            tuple or dict of torch.Tensor: batch data for system to fit

        """
        dataset = self.dataset
        if process_fn is not None:
            dataset = process_fn()
            logger.info('[Finish dataset process before batchify]')
        dataset = dataset[:ceil(len(dataset) * self.scale)]
        logger.debug(f'[Dataset size: {len(dataset)}]')

        batch_num = ceil(len(dataset) / batch_size)
        idx_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idx_list)

        for start_idx in tqdm(range(batch_num)):
            batch_idx = idx_list[start_idx * batch_size: (start_idx + 1) * batch_size]
            batch = [dataset[idx] for idx in batch_idx]
            
            neg_idx_list = self.neg_idx(start_idx, batch_num)
            neg_batch_idx = [idx_list[x * batch_size: (x + 1) * batch_size] for x in neg_idx_list]
            neg_batches = [[dataset[y] for y in x] for x in neg_batch_idx]
            yield batch_fn(batch, neg_batches)

    def neg_idx(self, start_idx, batch_num, neg_samples=5):
        try:
            neg_idx = np.random.randint(0, batch_num-2, size=neg_samples)
        except:
            neg_idx = np.random.randint(0,1,size=5)
        if start_idx in neg_idx:
            neg_idx = [(x+1)%(batch_num-2) for x in neg_idx]
        return neg_idx
    
    def get_pretrain_conv_data(self, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(self.pretrain_conv_batchify, batch_size, shuffle, self.pretrain_conv_process_fn)
    
    def get_conv_data(self, batch_size, shuffle=True):
        """get_data wrapper for conversation.

        You can implement your own process_fn in ``conv_process_fn``, batch_fn in ``conv_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for conversation.

        """
        return self.get_data(self.conv_batchify, batch_size, shuffle, self.conv_process_fn)

    def get_rec_data(self, batch_size, shuffle=True):
        """get_data wrapper for recommendation.

        You can implement your own process_fn in ``rec_process_fn``, batch_fn in ``rec_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for recommendation.

        """
        return self.get_data(self.rec_batchify, batch_size, shuffle, self.rec_process_fn)

    def get_policy_data(self, batch_size, shuffle=True):
        """get_data wrapper for policy.

        You can implement your own process_fn in ``self.policy_process_fn``, batch_fn in ``policy_batchify``.

        Args:
            batch_size (int):
            shuffle (bool, optional): Defaults to True.

        Yields:
            tuple or dict of torch.Tensor: batch data for policy.

        """
        return self.get_data(self.policy_batchify, batch_size, shuffle, self.policy_process_fn)

    def pretrain_conv_process_fn(self):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset
    
    def pretrain_conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError('dataloader must implement pretrain_conv_batchify() method')
    
    def conv_process_fn(self):
        """Process whole data for conversation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def conv_batchify(self, batch):
        """batchify data for conversation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train conversation part.
        """
        raise NotImplementedError('dataloader must implement conv_batchify() method')

    def rec_process_fn(self):
        """Process whole data for recommendation before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def rec_batchify(self, batch):
        """batchify data for recommendation after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train recommendation part.
        """
        raise NotImplementedError('dataloader must implement rec_batchify() method')

    def policy_process_fn(self):
        """Process whole data for policy before batch_fn.

        Returns:
            processed dataset. Defaults to return the same as `self.dataset`.

        """
        return self.dataset

    def policy_batchify(self, batch):
        """batchify data for policy after process.

        Args:
            batch (list): processed batch dataset.

        Returns:
            batch data for the system to train policy part.
        """
        raise NotImplementedError('dataloader must implement policy_batchify() method')

    def retain_recommender_target(self):
        """keep data whose role is recommender.

        Returns:
            Recommender part of ``self.dataset``.

        """
        dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                dataset.append(conv_dict)
        return dataset

    def rec_interact(self, data):
        """process user input data for system to recommend.

        Args:
            data: user input data.

        Returns:
            data for system to recommend.
        """
        pass

    def conv_interact(self, data):
        """Process user input data for system to converse.

        Args:
            data: user input data.

        Returns:
            data for system in converse.
        """
        pass
