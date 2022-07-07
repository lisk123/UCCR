import torch
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class UCCRSystem(BaseSystem):
    """This is the system for UCCR model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.

        """
        super(UCCRSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.pretrain_conv_optim_opt = self.opt['conv_pretrain']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        #self.conv_pretrain_epoch = self.pretrain_conv_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        #self.pretrain_conv_batch_size = self.pretrain_conv_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        
        self.tr_his_words_reps = []
        self.tr_his_entities_reps = []
        self.tr_word_attn_rep = []
        self.tr_entity_attn_rep = []
        self.tr_user_id = []

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch_all, stage, mode):
        if stage != 'conv':
            if stage == 'pretrain_conv':
                batch = batch_all
                batch = [ele.to(self.device) if type(ele)!=list else [x.to(self.device) for x in ele] for ele in batch]
            else:
                batch = batch_all[:-3]
                neg_batches = batch_all[-3:]
                batch = [ele.to(self.device) if type(ele)!=list else [x.to(self.device) for x in ele] for ele in batch]
                neg_batches = [[x.to(self.device) for x in ele] for ele in neg_batches]
        else:
            batch = [ele.to(self.device) if type(ele)!=list else [x.to(self.device) for x in ele] for ele in batch_all]
            #batch = [ele.to(self.device) for ele in batch]
        
        if stage == 'pretrain':
            pos_loss, contrastive_loss = self.model.pretrain_infomax(batch, neg_batches)
            if contrastive_loss:
                self.backward(contrastive_loss)
                contrastive_loss = contrastive_loss.item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(contrastive_loss))
        elif stage == 'rec':
            if self.tr_his_words_reps == []:
                for batch_all_tr in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=True):
                    batch_tr = batch_all_tr[:-3]
                    neg_batches_tr = batch_all_tr[-3:]
                    batch_tr = [ele.to(self.device) if type(ele)!=list else [x.to(self.device) for x in ele] for ele in batch_tr]
                    neg_batches_tr = [[x.to(self.device) for x in ele] for ele in neg_batches_tr]
                    cur_his_words_reps, cur_his_entities_reps, cur_his_items_reps, cur_word_attn_rep, cur_entity_attn_rep, cur_user_id = self.model.get_tr_history_reps(batch_tr, neg_batches_tr, mode)
                    if self.tr_his_words_reps == []:
                        self.tr_his_words_reps = cur_his_words_reps
                        self.tr_his_entities_reps = cur_his_entities_reps
                        self.tr_his_items_reps = cur_his_items_reps
                        self.tr_word_attn_rep = cur_word_attn_rep
                        self.tr_entity_attn_rep = cur_entity_attn_rep
                        self.tr_user_id = cur_user_id
                    else:
                        self.tr_his_words_reps = torch.cat([self.tr_his_words_reps, cur_his_words_reps], dim=0)
                        self.tr_his_entities_reps = torch.cat([self.tr_his_entities_reps, cur_his_entities_reps], dim=0)
                        self.tr_his_items_reps = torch.cat([self.tr_his_items_reps, cur_his_items_reps], dim=0)
                        self.tr_word_attn_rep = torch.cat([self.tr_word_attn_rep, cur_word_attn_rep], dim=0)
                        self.tr_entity_attn_rep = torch.cat([self.tr_entity_attn_rep, cur_entity_attn_rep], dim=0)
                        self.tr_user_id = torch.cat([self.tr_user_id, cur_user_id], dim=0)
                
                self.tr_his_words_reps = self.tr_his_words_reps.cuda()
                self.tr_his_entities_reps = self.tr_his_entities_reps.cuda()
                self.tr_his_items_reps = self.tr_his_items_reps.cuda()
                self.tr_word_attn_rep = self.tr_word_attn_rep.cuda()
                self.tr_entity_attn_rep = self.tr_entity_attn_rep.cuda()
                self.tr_user_id = self.tr_user_id.cuda()

            if mode != 'train':
                rec_loss, info_loss, rec_predict = self.model.recommend_test(batch, neg_batches, self.tr_his_words_reps, self.tr_his_entities_reps, self.tr_his_items_reps, self.tr_word_attn_rep, self.tr_entity_attn_rep, self.tr_user_id, mode)
            else:
                rec_loss, info_loss, rec_predict = self.model.recommend_training(batch, neg_batches, self.tr_his_words_reps, self.tr_his_entities_reps, self.tr_his_items_reps, self.tr_word_attn_rep, self.tr_entity_attn_rep, self.tr_user_id, mode)
                
            if mode == 'train':
                assert info_loss != None
            else:
                assert info_loss == None
            
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss)
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'pretrain_conv':
            loss = self.model.pretrain_converse(batch, mode)
            if loss:
                self.backward(loss)
                loss = loss.item()
                self.evaluator.optim_metrics.add("loss", AverageMetric(loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.converse(batch, mode)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.converse(batch, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            #import pdb
            #pdb.set_trace()
            for batch_all in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=True):
                self.step(batch_all, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.tr_his_words_reps = []
            self.tr_his_entities_reps = []
            self.tr_word_attn_rep = []
            self.tr_entity_attn_rep = []
            self.tr_user_id = []
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=True):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report()
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size*2, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report()
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                
                ################################### Test!
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_rec_data(self.rec_batch_size*2, shuffle=False):
                    self.step(batch, stage='rec', mode='test')
                self.evaluator.report()
                
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report()
            
    def train_conversation(self):
        self.model.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report()
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report()
                
                self.evaluator.reset_metrics()
                for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='test')
                self.evaluator.report()
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report()

    def eval_system(self):
        logger.info('[Valid]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.valid_dataloader.get_rec_data(1, shuffle=False):
                self.step(batch, stage='rec', mode='val')
            self.evaluator.report()
            # early stop
            metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']

            ################################### Test!
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(1, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report()
    
    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass
