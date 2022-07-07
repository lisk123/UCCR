# @Time   : 2020/12/4
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/12/6, 2021/1/2, 2020/12/19
# @Author : Kun Zhou, Xiaolei Wang, Yuanhang Zhou
# @Email  : francis_kun_zhou@163.com, sdzyh002@gmail

r"""
TGReDial
========
References:
    Zhou, Kun, et al. `"Towards Topic-Guided Conversational Recommender System."`_ in COLING 2020.

.. _`"Towards Topic-Guided Conversational Recommender System."`:
   https://www.aclweb.org/anthology/2020.coling-main.365/

"""

import json
import os
from collections import defaultdict
from copy import copy

from loguru import logger
from tqdm import tqdm

from crslab.config import DATASET_PATH
from crslab.data.dataset.base import BaseDataset
from .resources import resources

from copy import deepcopy


class TGReDialDataset(BaseDataset):
    """

    Attributes:
        train_data: train dataset.
        valid_data: valid dataset.
        test_data: test dataset.
        vocab (dict): ::

            {
                'tok2ind': map from token to index,
                'ind2tok': map from index to token,
                'topic2ind': map from topic to index,
                'ind2topic': map from index to topic,
                'entity2id': map from entity to index,
                'id2entity': map from index to entity,
                'word2id': map from word to index,
                'vocab_size': len(self.tok2ind),
                'n_topic': len(self.topic2ind) + 1,
                'n_entity': max(self.entity2id.values()) + 1,
                'n_word': max(self.word2id.values()) + 1,
            }

    Notes:
        ``'unk'`` and ``'pad_topic'`` must be specified in ``'special_token_idx'`` in ``resources.py``.

    """

    def __init__(self, opt, tokenize, restore=False, save=False):
        """Specify tokenized resource and init base dataset.

        Args:
            opt (Config or dict): config for dataset or the whole system.
            tokenize (str): how to tokenize dataset.
            restore (bool): whether to restore saved dataset which has been processed. Defaults to False.
            save (bool): whether to save dataset after processing. Defaults to False.

        """
        resource = resources[tokenize]
        self.special_token_idx = resource['special_token_idx']
        self.unk_token_idx = self.special_token_idx['unk']
        self.pad_topic_idx = self.special_token_idx['pad_topic']
        dpath = os.path.join(DATASET_PATH, 'tgredial', tokenize)
        super().__init__(opt, dpath, resource, restore, save)

    def _load_data(self):
        train_data, valid_data, test_data = self._load_raw_data()
        self._load_vocab()
        self._load_other_data()

        vocab = {
            'tok2ind': self.tok2ind,
            'ind2tok': self.ind2tok,
            'topic2ind': self.topic2ind,
            'ind2topic': self.ind2topic,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'word2id': self.word2id,
            'vocab_size': len(self.tok2ind),
            'n_topic': len(self.topic2ind) + 1,
            'n_entity': self.n_entity,
            'n_word': self.n_word,
        }
        vocab.update(self.special_token_idx)

        return train_data, valid_data, test_data, vocab

    def _load_raw_data(self):
        # load train/valid/test data
        with open(os.path.join(self.dpath, 'choiced_last_two_train_data.json'), 'r', encoding='utf-8') as f:
            train_data = json.load(f)
            logger.debug(f"[Load train data from {os.path.join(self.dpath, 'choiced_last_two_train_data.json')}]")
        with open(os.path.join(self.dpath, 'choiced_last_two_valid_data.json'), 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
            logger.debug(f"[Load valid data from {os.path.join(self.dpath, 'choiced_last_two_valid_data.json')}]")
        with open(os.path.join(self.dpath, 'choiced_last_two_test_data.json'), 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            logger.debug(f"[Load test data from {os.path.join(self.dpath, 'choiced_last_two_test_data.json')}]")

        return train_data, valid_data, test_data

    def _load_vocab(self):
        self.tok2ind = json.load(open(os.path.join(self.dpath, 'token2id.json'), 'r', encoding='utf-8'))
        self.ind2tok = {idx: word for word, idx in self.tok2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'token2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.tok2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2tok)}]")

        self.topic2ind = json.load(open(os.path.join(self.dpath, 'topic2id.json'), 'r', encoding='utf-8'))
        self.ind2topic = {idx: word for word, idx in self.topic2ind.items()}

        logger.debug(f"[Load vocab from {os.path.join(self.dpath, 'topic2id.json')}]")
        logger.debug(f"[The size of token2index dictionary is {len(self.topic2ind)}]")
        logger.debug(f"[The size of index2token dictionary is {len(self.ind2topic)}]")

    def _load_other_data(self):
        # cn-dbpedia
        self.entity2id = json.load(
            open(os.path.join(self.dpath, 'entity2id.json'), encoding='utf-8'))  # {entity: entity_id}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}
        self.n_entity = max(self.entity2id.values()) + 1
        # {head_entity_id: [(relation_id, tail_entity_id)]}
        self.entity_kg = open(os.path.join(self.dpath, 'cn-dbpedia.txt'), encoding='utf-8')
        logger.debug(
            f"[Load entity dictionary and KG from {os.path.join(self.dpath, 'entity2id.json')} and {os.path.join(self.dpath, 'cn-dbpedia.txt')}]")

        # hownet
        # {concept: concept_id}
        self.word2id = json.load(open(os.path.join(self.dpath, 'word2id.json'), 'r', encoding='utf-8'))
        self.n_word = max(self.word2id.values()) + 1
        # {relation\t concept \t concept}
        self.word_kg = open(os.path.join(self.dpath, 'hownet.txt'), encoding='utf-8')
        logger.debug(
            f"[Load word dictionary and KG from {os.path.join(self.dpath, 'word2id.json')} and {os.path.join(self.dpath, 'hownet.txt')}]")

        # user interaction history dictionary
        self.conv2history = json.load(open(os.path.join(self.dpath, 'user2history.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user interaction history from {os.path.join(self.dpath, 'user2history.json')}]")

        # user profile
        self.user2profile = json.load(open(os.path.join(self.dpath, 'user2profile.json'), 'r', encoding='utf-8'))
        logger.debug(f"[Load user profile from {os.path.join(self.dpath, 'user2profile.json')}")

    def _data_preprocess(self, train_data, valid_data, test_data):
        
        all_datas = train_data + valid_data + test_data
        conv_id_list = []
        for aaa in all_datas:
            conv_id_list.append(aaa['conv_id'])
        import numpy as np
        sorted_id = np.array(conv_id_list).argsort()
        sorted_all_data = []
        for iid in sorted_id:
            sorted_all_data.append(all_datas[iid])

        processed_train_data = []
        processed_valid_data = []
        processed_test_data = []
        
        augmented_all_convs = [self._convert_to_id(conversation) for conversation in tqdm(sorted_all_data)]
        
        last_user = None
        last_user_mentioned_words_list = []
        last_user_mentioned_entities_list = []
        
        conv_id_2_real_conv = dict()
        current_user_interactions = []
        time_id = 0
        is_new_begin = 0
        
        for i in tqdm(range(len(sorted_all_data))):
            current_user = sorted_all_data[i]['user_id']
            current_conv = augmented_all_convs[i]
            current_conv_id = sorted_all_data[i]['conv_id']
            cur_aug_and_add_conv = self._augment_and_add(current_conv)
            current_conv_interaction = dict()
            if last_user:
                if last_user == current_user:
                    time_id += 1
                    assert last_user_mentioned_words_list != []
                    assert last_user_mentioned_entities_list != []
                    tmp1 = cur_aug_and_add_conv[-1]['context_entities']
                    tmp2 = cur_aug_and_add_conv[-1]['context_words']
                    for xx in cur_aug_and_add_conv:
                        #if xx['items'] != [] and xx['role'] == 'Recommender':
                        yyy = deepcopy(current_user_interactions)
                        xx['histories'] = yyy
                        #xx['context_entities'] = last_user_mentioned_entities_list[-1] + xx['context_entities']
                        #xx['context_words'] = last_user_mentioned_words_list[-1] + xx['context_words']
                else:
                    time_id = 0
                    last_user_mentioned_words_list = []
                    last_user_mentioned_entities_list = []
                    current_user_interactions = []
                    for xx in cur_aug_and_add_conv:
                        #if xx['items'] != [] and xx['role'] == 'Recommender':
                        yyy = deepcopy(current_user_interactions)
                        xx['histories'] = yyy
                    is_new_begin = 1
            else:
                assert current_user_interactions == []
                for xx in cur_aug_and_add_conv:
                    #if xx['items'] != [] and xx['role'] == 'Recommender':
                    yyy = deepcopy(current_user_interactions)
                    xx['histories'] = yyy
            
            for xx in cur_aug_and_add_conv:
                xx['time_id'] = time_id
                xx['user_id'] = int(current_user)
                if xx['items'] != [] and xx['role'] == 'Recommender':
                    if current_conv_interaction == dict():
                        current_conv_interaction['history_entity'] = []
                        current_conv_interaction['history_entities_pos'] = []
                        current_conv_interaction['history_word'] = []
                        current_conv_interaction['history_words_pos'] = []
                        current_conv_interaction['history_item'] = []
                    else:
                        current_conv_interaction['history_entity'].append(xx['context_entities'])
                        current_conv_interaction['history_entities_pos'].append(xx['context_entities_pos'])
                        current_conv_interaction['history_word'].append(xx['context_words'])
                        current_conv_interaction['history_words_pos'].append(xx['context_words_pos'])
                        current_conv_interaction['history_item'].append(xx['items'])
            if is_new_begin:
                is_new_begin=0
                current_user_interactions.append(current_conv_interaction)
            else:
                current_user_interactions.append(current_conv_interaction)
            
            conv_id_2_real_conv[current_conv_id] = cur_aug_and_add_conv
            
            if last_user and last_user == current_user:
                last_user_mentioned_words_list.append(tmp2)
                last_user_mentioned_entities_list.append(tmp1)
            else:
                last_user_mentioned_words_list.append(cur_aug_and_add_conv[-1]['context_words'])
                last_user_mentioned_entities_list.append(cur_aug_and_add_conv[-1]['context_entities'])
            
            last_user = current_user
        
        for tr in train_data:
            c_id = tr['conv_id']
            processed_train_data.extend(conv_id_2_real_conv[c_id])
        for va in valid_data:
            c_id = va['conv_id']
            processed_valid_data.extend(conv_id_2_real_conv[c_id])
        for te in test_data:
            c_id = te['conv_id']
            processed_test_data.extend(conv_id_2_real_conv[c_id])
        
        # check:
        processed_train_data_duizhao = self._raw_data_process(train_data)
        processed_valid_data_duizhao = self._raw_data_process(valid_data)
        processed_test_data_duizhao = self._raw_data_process(test_data)
        
        responses_ids_ptr = []
        responses_ids_duizhao_ptr = []
        for ptr in tqdm(processed_train_data):
            responses_ids_ptr.append(tuple(ptr['response']))
        for ptr_d in tqdm(processed_train_data_duizhao):
            responses_ids_duizhao_ptr.append(tuple(ptr_d['response']))
        assert set(responses_ids_ptr) == set(responses_ids_duizhao_ptr)
        
        responses_ids_ptr = []
        responses_ids_duizhao_ptr = []
        for ptr in tqdm(processed_valid_data):
            responses_ids_ptr.append(tuple(ptr['response']))
        for ptr_d in tqdm(processed_valid_data_duizhao):
            responses_ids_duizhao_ptr.append(tuple(ptr_d['response']))
        assert set(responses_ids_ptr) == set(responses_ids_duizhao_ptr)
        
        responses_ids_ptr = []
        responses_ids_duizhao_ptr = []
        for ptr in tqdm(processed_test_data):
            responses_ids_ptr.append(tuple(ptr['response']))
        for ptr_d in tqdm(processed_test_data_duizhao):
            responses_ids_duizhao_ptr.append(tuple(ptr_d['response']))
        assert set(responses_ids_ptr) == set(responses_ids_duizhao_ptr)
        
        processed_side_data = self._side_data_process()
        logger.debug("[Finish side data process]")
        return processed_train_data, processed_valid_data, processed_test_data, processed_side_data

    def _raw_data_process(self, raw_data):
        augmented_convs = [self._convert_to_id(conversation) for conversation in tqdm(raw_data)]
        augmented_conv_dicts = []
        for conv in tqdm(augmented_convs):
            augmented_conv_dicts.extend(self._augment_and_add(conv))
        return augmented_conv_dicts

    def _convert_to_id(self, conversation):
        augmented_convs = []
        last_role = None
        for utt in conversation['messages']:
            assert utt['role'] != last_role

            text_token_ids = [self.tok2ind.get(word, self.unk_token_idx) for word in utt["text"]]
            movie_ids = [self.entity2id[movie] for movie in utt['movie'] if movie in self.entity2id]
            entity_ids = [self.entity2id[entity] for entity in utt['entity'] if entity in self.entity2id]
            word_ids = [self.word2id[word] for word in utt['word'] if word in self.word2id]
            policy = []
            for action, kw in zip(utt['target'][1::2], utt['target'][2::2]):
                if kw is None or action == '推荐电影':
                    continue
                if isinstance(kw, str):
                    kw = [kw]
                kw = [self.topic2ind.get(k, self.pad_topic_idx) for k in kw]
                policy.append([action, kw])
            final_kws = [self.topic2ind[kw] if kw is not None else self.pad_topic_idx for kw in utt['final'][1]]
            final = [utt['final'][0], final_kws]
            conv_utt_id = str(conversation['conv_id']) + '/' + str(utt['local_id'])
            interaction_history = self.conv2history.get(conv_utt_id, [])
            user_profile = self.user2profile[conversation['user_id']]
            user_profile = [[self.tok2ind.get(token, self.unk_token_idx) for token in sent] for sent in user_profile]

            augmented_convs.append({
                "role": utt["role"],
                "text": text_token_ids,
                "entity": entity_ids,
                "movie": movie_ids,
                "word": word_ids,
                'policy': policy,
                'final': final,
                'interaction_history': interaction_history,
                'user_profile': user_profile
            })
            last_role = utt["role"]

        return augmented_convs

    def _augment_and_add(self, raw_conv_dict):
        augmented_conv_dicts = []
        context_tokens, context_entities, context_words, context_policy, context_items = [], [], [], [], []
        context_entities_pos, context_words_pos = [], []
        entity_set, word_set = set(), set()
        for i, conv in enumerate(raw_conv_dict):
            text_tokens, entities, movies, words, policies = conv["text"], conv["entity"], conv["movie"], conv["word"], \
                                                             conv['policy']
            if len(context_tokens) > 0:
                conv_dict = {
                    'role': conv['role'],
                    'user_profile': conv['user_profile'],
                    "context_tokens": copy(context_tokens),
                    "response": text_tokens,
                    "context_entities": copy(context_entities),
                    "context_entities_pos": copy(context_entities_pos),
                    "context_words": copy(context_words),
                    "context_words_pos": copy(context_words_pos),
                    'interaction_history': conv['interaction_history'],
                    'context_items': copy(context_items),
                    "items": movies,
                    'context_policy': copy(context_policy),
                    'target': policies,
                    'final': conv['final'],
                }
                augmented_conv_dicts.append(conv_dict)

            context_tokens.append(text_tokens)
            context_policy.append(policies)
            context_items += movies
            for entity in entities + movies:
                if entity not in entity_set:
                    entity_set.add(entity)
                    context_entities.append(entity)
                    context_entities_pos.append(i)
            for word in words:
                if word not in word_set:
                    word_set.add(word)
                    context_words.append(word)
                    context_words_pos.append(i)

        return augmented_conv_dicts

    def _side_data_process(self):
        processed_entity_kg = self._entity_kg_process()
        logger.debug("[Finish entity KG process]")
        processed_word_kg = self._word_kg_process()
        logger.debug("[Finish word KG process]")
        movie_entity_ids = json.load(open(os.path.join(self.dpath, 'movie_ids.json'), 'r', encoding='utf-8'))
        logger.debug('[Load movie entity ids]')

        side_data = {
            "entity_kg": processed_entity_kg,
            "word_kg": processed_word_kg,
            "item_entity_ids": movie_entity_ids,
        }
        return side_data

    def _entity_kg_process(self):
        edge_list = []  # [(entity, entity, relation)]
        for line in self.entity_kg:
            triple = line.strip().split('\t')
            e0 = self.entity2id[triple[0]]
            e1 = self.entity2id[triple[2]]
            r = triple[1]
            edge_list.append((e0, e1, r))
            edge_list.append((e1, e0, r))
            edge_list.append((e0, e0, 'SELF_LOOP'))
            if e1 != e0:
                edge_list.append((e1, e1, 'SELF_LOOP'))

        relation_cnt, relation2id, edges, entities = defaultdict(int), dict(), set(), set()
        for h, t, r in edge_list:
            relation_cnt[r] += 1
        for h, t, r in edge_list:
            if r not in relation2id:
                relation2id[r] = len(relation2id)
            edges.add((h, t, relation2id[r]))
            entities.add(self.id2entity[h])
            entities.add(self.id2entity[t])

        return {
            'edge': list(edges),
            'n_relation': len(relation2id),
            'entity': list(entities)
        }

    def _word_kg_process(self):
        edges = set()  # {(entity, entity)}
        entities = set()
        for line in self.word_kg:
            triple = line.strip().split('\t')
            entities.add(triple[0])
            entities.add(triple[2])
            e0 = self.word2id[triple[0]]
            e1 = self.word2id[triple[2]]
            edges.add((e0, e1))
            edges.add((e1, e0))
        # edge_set = [[co[0] for co in list(edges)], [co[1] for co in list(edges)]]
        return {
            'edge': list(edges),
            'entity': list(entities)
        }
