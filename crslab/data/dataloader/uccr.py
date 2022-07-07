# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from copy import deepcopy

import torch
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt


class UCCRDataLoader(BaseDataLoader):
    """Dataloader for model UCCR.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.
        - ``'word_truncate'``: the maximum length of mentioned words in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``
        - ``'pad_word'``

        the above values specify the id of needed special token.

        - ``'n_entity'``: the number of entities in the entity KG of dataset.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.n_entity = vocab['n_entity']
        self.n_word = vocab['n_word']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch, neg_batches):
        batch_context_entities = []
        batch_context_words = []
        batch_time_id = []
        batch_history_items = []
        batch_history_items_pos = []
        batch_h_words = []
        batch_h_words_pos = []
        batch_h_entities = []
        batch_h_entities_pos = []
        batch_h_items = []
        batch_user_id = []
        for kkk in range(len(batch)):
            conv_dict = batch[kkk]
            batch_user_id.append(conv_dict['user_id'])
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_time_id.append(conv_dict['time_id'])
            history_item_all = []
            history_item_pos_all = []
            h_word = []
            h_word_pos = []
            h_entity = []
            h_entity_pos = []
            h_item = []
            for i in range(len(conv_dict['histories'])):
                #import pdb
                #pdb.set_trace()
                xx = conv_dict['histories'][i]
                tt = [x[0] for x in xx['history_item']]
                h_word.extend(xx['history_word'])
                h_word_pos.extend(xx['history_words_pos'])
                h_entity.extend(xx['history_entity'])
                h_entity_pos.extend(xx['history_entities_pos'])
                h_item.extend(xx['history_item'])
                for _ in range(len(tt)):
                    history_item_pos_all.append(i)
                history_item_all.extend(tt)
            batch_history_items_pos.append(
                truncate(history_item_pos_all, self.entity_truncate, truncate_tail=False))
            batch_history_items.append(
                truncate(history_item_all, self.entity_truncate, truncate_tail=False))
            if h_word == []:
                batch_h_words.append([[]])
                batch_h_words_pos.append([[]])
            else:
                batch_h_words.append(
                    truncate(h_word, self.word_truncate, truncate_tail=False))
                batch_h_words_pos.append(
                    truncate(h_word_pos, self.word_truncate, truncate_tail=False))
            if h_entity == []:
                batch_h_entities.append([[]])
                batch_h_entities_pos.append([[]])
            else:
                batch_h_entities.append(
                    truncate(h_entity, self.word_truncate, truncate_tail=False))
                batch_h_entities_pos.append(
                    truncate(h_entity_pos, self.word_truncate, truncate_tail=False))
            if h_item == []:
                batch_h_items.append([[]])
            else:
                batch_h_items.append(
                    truncate(h_item, self.word_truncate, truncate_tail=False))
            '''
            batch_h_words[kkk].append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_h_entities[kkk].append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            '''

        
        padded_history_words = []
        padded_history_words_pos = []
        padded_history_entities = []
        padded_history_entities_pos = []
        padded_history_items = []
        for i in range(len(batch_h_words)):
            batch_c_words = batch_h_words[i]
            batch_c_words_pos = batch_h_words_pos[i]
            batch_c_entities = batch_h_entities[i]
            batch_c_entities_pos = batch_h_entities_pos[i]
            batch_c_items = batch_h_items[i]
            padded_history_words.append(padded_tensor(batch_c_words, self.pad_word_idx, pad_tail=False))
            padded_history_words_pos.append(padded_tensor(batch_c_words_pos, self.pad_word_idx, pad_tail=False))
            padded_history_entities.append(padded_tensor(batch_c_entities, self.pad_entity_idx, pad_tail=False))
            padded_history_entities_pos.append(padded_tensor(batch_c_entities_pos, self.pad_entity_idx, pad_tail=False))
            padded_history_items.append(padded_tensor(batch_c_items, self.pad_entity_idx, pad_tail=False))
        #batch_history_items = [x[0] for x in batch_history_items]
        neg_batches_context_entities = []
        neg_batches_context_words = []
        for nbs in neg_batches:
            cur_nbce = []
            cur_nbcw = []
            for conv in nbs:
                cur_nbce.append(
                    truncate(conv['context_entities'], self.entity_truncate, truncate_tail=False))
                cur_nbcw.append(truncate(conv['context_words'], self.word_truncate, truncate_tail=False))
            neg_batches_context_entities.append(cur_nbce)
            neg_batches_context_words.append(cur_nbcw)

        return_nbcei = [padded_tensor(x, self.pad_entity_idx, pad_tail=False) for x in neg_batches_context_entities]
        return_nbcw = [padded_tensor(x, self.pad_word_idx, pad_tail=False) for x in neg_batches_context_words]
        return_nbcel = [get_onehot(x, self.n_entity) for x in neg_batches_context_entities]
        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_history_items, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_history_items_pos, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                get_onehot(batch_context_words, self.n_word),
                get_onehot(batch_history_items, self.n_entity),
                padded_history_words,
                padded_history_words_pos,
                padded_history_entities,
                padded_history_entities_pos,
                padded_history_items,
                torch.tensor(batch_user_id, dtype=torch.long),
                torch.tensor(batch_time_id, dtype=torch.long),
                return_nbcei,
                return_nbcw,
                return_nbcel)

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch, neg_batches):
        batch_context_entities = []
        batch_context_words = []
        batch_item = []
        batch_time_id = []
        batch_history_items = []
        batch_history_items_pos = []
        batch_h_words = []
        batch_h_words_pos = []
        batch_h_entities = []
        batch_h_entities_pos = []
        batch_h_items = []
        batch_user_id = []
        for kkk in range(len(batch)):
            conv_dict = batch[kkk]
            batch_user_id.append(conv_dict['user_id'])
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_item.append(conv_dict['item'])
            batch_time_id.append(conv_dict['time_id'])
            history_item_all = []
            history_item_pos_all = []
            h_word = []
            h_word_pos = []
            h_entity = []
            h_entity_pos = []
            h_item = []
            for i in range(len(conv_dict['histories'])):
                #import pdb
                #pdb.set_trace()
                xx = conv_dict['histories'][i]
                tt = [x[0] for x in xx['history_item']]
                h_word.extend(xx['history_word'])
                h_word_pos.extend(xx['history_words_pos'])
                h_entity.extend(xx['history_entity'])
                h_entity_pos.extend(xx['history_entities_pos'])
                h_item.extend(xx['history_item'])
                for _ in range(len(tt)):
                    history_item_pos_all.append(i)
                history_item_all.extend(tt)
            batch_history_items_pos.append(
                truncate(history_item_pos_all, self.entity_truncate, truncate_tail=False))
            batch_history_items.append(
                truncate(history_item_all, self.entity_truncate, truncate_tail=False))
            if h_word == []:
                batch_h_words.append([[]])
                batch_h_words_pos.append([[]])
            else:
                batch_h_words.append(
                    truncate(h_word, self.word_truncate, truncate_tail=False))
                batch_h_words_pos.append(
                    truncate(h_word_pos, self.word_truncate, truncate_tail=False))
            if h_entity == []:
                batch_h_entities.append([[]])
                batch_h_entities_pos.append([[]])
            else:
                batch_h_entities.append(
                    truncate(h_entity, self.word_truncate, truncate_tail=False))
                batch_h_entities_pos.append(
                    truncate(h_entity_pos, self.word_truncate, truncate_tail=False))
            if h_item == []:
                batch_h_items.append([[]])
            else:
                batch_h_items.append(
                    truncate(h_item, self.word_truncate, truncate_tail=False))
            '''
            batch_h_words[kkk].append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_h_entities[kkk].append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            '''
        
        padded_history_words = []
        padded_history_words_pos = []
        padded_history_entities = []
        padded_history_entities_pos = []
        padded_history_items = []
        for i in range(len(batch_h_words)):
            batch_c_words = batch_h_words[i]
            batch_c_words_pos = batch_h_words_pos[i]
            batch_c_entities = batch_h_entities[i]
            batch_c_entities_pos = batch_h_entities_pos[i]
            batch_c_items = batch_h_items[i]
            padded_history_words.append(padded_tensor(batch_c_words, self.pad_word_idx, pad_tail=False))
            padded_history_words_pos.append(padded_tensor(batch_c_words_pos, self.pad_word_idx, pad_tail=False))
            padded_history_entities.append(padded_tensor(batch_c_entities, self.pad_entity_idx, pad_tail=False))
            padded_history_entities_pos.append(padded_tensor(batch_c_entities_pos, self.pad_entity_idx, pad_tail=False))
            padded_history_items.append(padded_tensor(batch_c_items, self.pad_entity_idx, pad_tail=False))
        neg_batches_context_entities = []
        neg_batches_context_words = []
        for nbs in neg_batches:
            cur_nbce = []
            cur_nbcw = []
            for conv in nbs:
                cur_nbce.append(
                    truncate(conv['context_entities'], self.entity_truncate, truncate_tail=False))
                cur_nbcw.append(truncate(conv['context_words'], self.word_truncate, truncate_tail=False))
            neg_batches_context_entities.append(cur_nbce)
            neg_batches_context_words.append(cur_nbcw)

        return_nbcei = [padded_tensor(x, self.pad_entity_idx, pad_tail=False) for x in neg_batches_context_entities]
        return_nbcw = [padded_tensor(x, self.pad_word_idx, pad_tail=False) for x in neg_batches_context_words]
        return_nbcel = [get_onehot(x, self.n_entity) for x in neg_batches_context_entities]
        
        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_history_items, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_history_items_pos, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                get_onehot(batch_context_words, self.n_word),
                get_onehot(batch_history_items, self.n_entity),
                padded_history_words,
                padded_history_words_pos,
                padded_history_entities,
                padded_history_entities_pos,
                padded_history_items,
                torch.tensor(batch_user_id, dtype=torch.long),
                torch.tensor(batch_time_id, dtype=torch.long),
                torch.tensor(batch_item, dtype=torch.long),
                return_nbcei,
                return_nbcw,
                return_nbcel)

    def pretrain_conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def pretrain_conv_batchify(self, batch, neg_batches):
        batch_context_tokens = []
        batch_context_masked_tokens_seq = []
        batch_context_masked_tokens_label = []
        batch_responses = []
        batch_labels = []

        for i in range(len(batch)):
            import numpy as np
            import copy
            conv_dict = batch[i]
            tokens = copy.deepcopy(merge_utt(conv_dict['context_tokens']))
            N_tokens = len(tokens)
            pos = np.random.randint(N_tokens)
            to_predict_token = tokens[pos]
            tokens[pos] = 0
            batch_context_masked_tokens_seq.append(truncate(tokens, self.context_truncate, truncate_tail=False))
            batch_context_masked_tokens_label.append(to_predict_token)
            
            batch_context_tokens.append(truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            choice = np.random.randint(2)
            if choice:
                neg_response = neg_batches[0][i]['response']
                batch_responses.append(add_start_end_token_idx(truncate(neg_response, self.response_truncate - 2), start_token_idx=self.start_token_idx, end_token_idx=self.end_token_idx))
                batch_labels.append(0.)
            else:
                batch_responses.append(add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2), start_token_idx=self.start_token_idx, end_token_idx=self.end_token_idx))
                batch_labels.append(1.)

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_masked_tokens_seq, self.pad_token_idx, pad_tail=False),
                torch.tensor(batch_context_masked_tokens_label),
                padded_tensor(batch_responses, self.pad_token_idx),
                torch.tensor(batch_labels))
    
    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch, neg_batches):
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        batch_history_items = []
        batch_h_words = []
        batch_h_entities = []
        batch_h_items = []
        batch_user_id = []
        for kkk in range(len(batch)):
            conv_dict = batch[kkk]
            batch_user_id.append(conv_dict['user_id'])
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))
            history_item_all = []
            h_word = []
            h_entity = []
            h_item = []
            for i in range(len(conv_dict['histories'])):
                #import pdb
                #pdb.set_trace()
                xx = conv_dict['histories'][i]
                tt = [x[0] for x in xx['history_item']]
                h_word.extend(xx['history_word'])
                h_entity.extend(xx['history_entity'])
                h_item.extend(xx['history_item'])
                history_item_all.extend(tt)
            batch_history_items.append(
                truncate(history_item_all, self.entity_truncate, truncate_tail=False))
            if h_word == []:
                batch_h_words.append([[]])
            else:
                batch_h_words.append(
                    truncate(h_word, self.word_truncate, truncate_tail=False))
            if h_entity == []:
                batch_h_entities.append([[]])
            else:
                batch_h_entities.append(
                    truncate(h_entity, self.word_truncate, truncate_tail=False))
            if h_item == []:
                batch_h_items.append([[]])
            else:
                batch_h_items.append(
                    truncate(h_item, self.word_truncate, truncate_tail=False))
            '''
            batch_h_words[kkk].append(truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_h_entities[kkk].append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            '''

        
        padded_history_words = []
        padded_history_entities = []
        padded_history_items = []
        for i in range(len(batch_h_words)):
            batch_c_words = batch_h_words[i]
            batch_c_entities = batch_h_entities[i]
            batch_c_items = batch_h_items[i]
            padded_history_words.append(padded_tensor(batch_c_words, self.pad_word_idx, pad_tail=False))
            padded_history_entities.append(padded_tensor(batch_c_entities, self.pad_entity_idx, pad_tail=False))
            padded_history_items.append(padded_tensor(batch_c_items, self.pad_entity_idx, pad_tail=False))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_history_items, self.pad_entity_idx, pad_tail=False),
                padded_history_words,
                padded_history_entities,
                padded_history_items,
                torch.tensor(batch_user_id, dtype=torch.long),
                padded_tensor(batch_response, self.pad_token_idx))

    def policy_batchify(self, *args, **kwargs):
        pass
