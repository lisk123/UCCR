import os

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv

from crslab.config import MODEL_PATH
from crslab.model.base import BaseModel
from crslab.model.utils.functions import edge_to_pyg_format
from crslab.model.utils.modules.attention import SelfAttentionSeq
from crslab.model.utils.modules.transformer import TransformerEncoder
from .modules import GateLayer, TransformerDecoderKG, GateLayer_4_eles, GateLayer_5_eles, GateLayer_3_eles
from .resources import resources

class UCCRModel(BaseModel):
    """

    Attributes:
        vocab_size: A integer indicating the vocabulary size.
        pad_token_idx: A integer indicating the id of padding token.
        start_token_idx: A integer indicating the id of start token.
        end_token_idx: A integer indicating the id of end token.
        token_emb_dim: A integer indicating the dimension of token embedding layer.
        pretrain_embedding: A string indicating the path of pretrained embedding.
        n_word: A integer indicating the number of words.
        n_entity: A integer indicating the number of entities.
        pad_word_idx: A integer indicating the id of word padding.
        pad_entity_idx: A integer indicating the id of entity padding.
        num_bases: A integer indicating the number of bases.
        kg_emb_dim: A integer indicating the dimension of kg embedding.
        n_heads: A integer indicating the number of heads.
        n_layers: A integer indicating the number of layer.
        ffn_size: A integer indicating the size of ffn hidden.
        dropout: A float indicating the dropout rate.
        attention_dropout: A integer indicating the dropout rate of attention layer.
        relu_dropout: A integer indicating the dropout rate of relu layer.
        learn_positional_embeddings: A boolean indicating if we learn the positional embedding.
        embeddings_scale: A boolean indicating if we use the embeddings scale.
        reduction: A boolean indicating if we use the reduction.
        n_positions: A integer indicating the number of position.
        response_truncate = A integer indicating the longest length for response generation.
        pretrained_embedding: A string indicating the path of pretrained embedding.

    """

    def __init__(self, opt, device, vocab, side_data):
        """

        Args:
            opt (dict): A dictionary record the hyper parameters.
            device (torch.device): A variable indicating which device to place the data and model.
            vocab (dict): A dictionary record the vocabulary information.
            side_data (dict): A dictionary record the side data.

        """
        # vocab
        self.vocab_size = vocab['vocab_size']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.token_emb_dim = opt['token_emb_dim']
        self.pretrained_embedding = side_data.get('embedding', None)
        # kg
        self.n_word = vocab['n_word']
        self.n_entity = vocab['n_entity']
        self.pad_word_idx = vocab['pad_word']
        self.pad_entity_idx = vocab['pad_entity']
        entity_kg = side_data['entity_kg']
        self.n_relation = entity_kg['n_relation']
        entity_edges = entity_kg['edge']
        self.entity_edge_idx, self.entity_edge_type = edge_to_pyg_format(entity_edges, 'RGCN')
        self.entity_edge_idx = self.entity_edge_idx.to(device)
        self.entity_edge_type = self.entity_edge_type.to(device)
        word_edges = side_data['word_kg']['edge']
        self.word_edges = edge_to_pyg_format(word_edges, 'GCN').to(device)
        self.num_bases = opt['num_bases']
        self.kg_emb_dim = opt['kg_emb_dim']
        # transformer
        self.n_heads = opt['n_heads']
        self.n_layers = opt['n_layers']
        self.ffn_size = opt['ffn_size']
        self.dropout = opt['dropout']
        self.attention_dropout = opt['attention_dropout']
        self.relu_dropout = opt['relu_dropout']
        self.learn_positional_embeddings = opt['learn_positional_embeddings']
        self.embeddings_scale = opt['embeddings_scale']
        self.reduction = opt['reduction']
        self.n_positions = opt['n_positions']
        self.response_truncate = opt.get('response_truncate', 20)
        # copy mask
        self.dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "kgsf", self.dataset)
        resource = resources[self.dataset]
        super(UCCRModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        self._init_embeddings()
        self._build_kg_layer()
        self._build_infomax_layer()
        self._build_recommendation_layer()
        self._build_conversation_layer()

    def _init_embeddings(self):
        if self.pretrained_embedding is not None:
            self.token_embedding = nn.Embedding.from_pretrained(
                torch.as_tensor(self.pretrained_embedding, dtype=torch.float), freeze=False,
                padding_idx=self.pad_token_idx)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, self.token_emb_dim, self.pad_token_idx)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
            nn.init.constant_(self.token_embedding.weight[self.pad_token_idx], 0)

        self.word_kg_embedding = nn.Embedding(self.n_word, self.kg_emb_dim, self.pad_word_idx)
        nn.init.normal_(self.word_kg_embedding.weight, mean=0, std=self.kg_emb_dim ** -0.5)
        nn.init.constant_(self.word_kg_embedding.weight[self.pad_word_idx], 0)

        self.time_id_embedding = nn.Embedding(115, self.kg_emb_dim)
        self.time_id_linear = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)

        logger.debug('[Finish init embeddings]')

    def _build_kg_layer(self):
        # db encoder
        self.entity_encoder = RGCNConv(self.n_entity, self.kg_emb_dim, self.n_relation, self.num_bases)
        self.entity_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # concept encoder
        self.word_encoder = GCNConv(self.kg_emb_dim, self.kg_emb_dim)
        self.word_self_attn = SelfAttentionSeq(self.kg_emb_dim, self.kg_emb_dim)

        # gate mechanism
        self.gate_layer = GateLayer(self.kg_emb_dim)
        self.gate_layer2 = GateLayer(self.kg_emb_dim)
        self.gate_layer_3_eles = GateLayer_3_eles(self.kg_emb_dim)
        self.gate_layer_4_eles = GateLayer_4_eles(self.kg_emb_dim)
        self.gate_layer_5_eles = GateLayer_5_eles(self.kg_emb_dim)
        self.word_lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.1)
        self.entity_lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, bidirectional=False, dropout=0.1)
        self.word_weight_matrix = nn.Linear(128, 128)
        self.entity_weight_matrix = nn.Linear(128, 128)
        self.state_dim_map_double = nn.Linear(128*2, 128)
        self.state_dim_map = nn.Linear(128, 128)
        self.simi_weight = nn.Linear(128, 128)
        self.user_his_item_rep_map = nn.Linear(128*2, 128)
        self.rec_norm = nn.Linear(128, 128)
        self.combine_linear = nn.Linear(128*2, 128)
        self.sim_proj = nn.Linear(128*2, 1)
        self.past_cur_rep_sigma = nn.Linear(128*2, 1)
        self.user_rep_to_vocab = nn.Linear(128, self.vocab_size)
        self.past_cur_weight_matrix = nn.Linear(128, 128)
        #self.gate_layer3 = GateLayer(self.kg_emb_dim)

        logger.debug('[Finish build kg layer]')

    def _build_infomax_layer(self):
        #self.word_infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        #self.entity_infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.word_infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.entity_infomax_norm = nn.Linear(self.kg_emb_dim, self.kg_emb_dim)
        self.infomax_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.infomax_bias_word = nn.Linear(self.kg_emb_dim, self.n_word)
        self.infomax_loss = nn.MSELoss(reduction='sum')

        logger.debug('[Finish build infomax layer]')

    def _build_recommendation_layer(self):
        self.rec_bias = nn.Linear(self.kg_emb_dim, self.n_entity)
        self.rec_loss = nn.CrossEntropyLoss()

        logger.debug('[Finish build rec layer]')

    def _build_conversation_layer(self):
        self.register_buffer('START', torch.tensor([self.start_token_idx], dtype=torch.long))
        self.conv_encoder = TransformerEncoder(
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            embedding_size=self.token_emb_dim,
            ffn_size=self.ffn_size,
            vocabulary_size=self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            padding_idx=self.pad_token_idx,
            learn_positional_embeddings=self.learn_positional_embeddings,
            embeddings_scale=self.embeddings_scale,
            reduction=self.reduction,
            n_positions=self.n_positions,
        )

        self.conv_entity_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_entity_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)
        self.conv_word_attn_norm = nn.Linear(self.kg_emb_dim, self.ffn_size)

        self.copy_norm = nn.Linear(self.ffn_size * 3, self.token_emb_dim)
        self.copy_output = nn.Linear(self.token_emb_dim, self.vocab_size)
        self.copy_mask = torch.as_tensor(np.load(os.path.join(self.dpath, "copy_mask.npy")).astype(bool),
                                         device=self.device)

        self.conv_decoder = TransformerDecoderKG(
            self.n_heads, self.n_layers, self.token_emb_dim, self.ffn_size, self.vocab_size,
            embedding=self.token_embedding,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            relu_dropout=self.relu_dropout,
            embeddings_scale=self.embeddings_scale,
            learn_positional_embeddings=self.learn_positional_embeddings,
            padding_idx=self.pad_token_idx,
            n_positions=self.n_positions
        )
        self.conv_loss = nn.CrossEntropyLoss(ignore_index=self.pad_token_idx)
        

        logger.debug('[Finish build conv layer]')
        
    def compute_barrow_loss(self, view_1_rep, view_2_rep, mu):
        cov_matrix_up = torch.matmul(view_1_rep, view_2_rep.t())
        bs = view_1_rep.shape[0]
        words_down = (view_1_rep * view_1_rep).sum(dim=1).view(bs, 1)
        entities_down = (view_2_rep * view_2_rep).sum(dim=1).view(1, bs)
        words_down = words_down.expand(bs, bs)
        entities_down = entities_down.expand(bs, bs)
        cov_matrix_down = torch.sqrt(words_down * entities_down)
        cov_matrix = cov_matrix_up / cov_matrix_down
        mask_part1 = torch.eye(bs).to(self.device)
        mask_part2 = torch.ones((bs, bs)).to(self.device) - mask_part1
        
        loss_part1 = ((mask_part1 - cov_matrix).diag() * (mask_part1 - cov_matrix).diag()).sum()
        loss_part2 = ((mask_part2 * cov_matrix) * (mask_part2 * cov_matrix)).sum()
        loss = loss_part1 + mu * loss_part2
        
        return loss_part1, loss_part2, loss
    
    def get_user_historical_info(self, words, entity_indexs, history_items, h_words, h_words_pos, h_entities, h_items, word_graph_representations, entity_graph_representations, entity_attn_rep, word_attn_rep, time_id, get_tr_his_reps=False):
        cur_fusion_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        
        history_words_reps = []
        
        for ii in range(len(h_words)):
            hw = h_words[ii]
            hw_local_pos = h_words_pos[ii]
            try:
                yyy=hw.item()
                hw = words[ii].view(1,-1)
                if self.dataset == 'ReDial' and int((hw==0).sum().cpu().numpy())==hw.shape[1]:
                    c_word_reps = entity_graph_representations[hw]
                    c_w_padding_mask = hw.eq(self.pad_entity_idx)
                else:
                    c_word_reps = word_graph_representations[hw]
                    c_w_padding_mask = hw.eq(self.pad_word_idx)
                c_word_attn_rep = self.word_self_attn(c_word_reps, c_w_padding_mask)
                
                simi = F.softmax(torch.matmul(self.word_weight_matrix(c_word_attn_rep), word_attn_rep[ii,:])*10)
                c_word_true_rep = (simi.view(-1, 1) * c_word_attn_rep).sum(dim=0)
                history_words_reps.append(c_word_true_rep)
            except:
                xxxx=1
                c_word_reps = word_graph_representations[hw]
                c_w_padding_mask = hw.eq(self.pad_word_idx)
                local_turn_attn = F.softmax(hw_local_pos.float() + (-1e30*c_w_padding_mask))
                c_word_attn_rep = local_turn_attn.unsqueeze(dim=2)*c_word_reps
                c_word_attn_rep = c_word_attn_rep.sum(dim=1)
                global_turn_attn = []
                if self.dataset == 'ReDial':
                    for xx in range(hw.shape[0]):
                        global_turn_attn.append(xx+1)
                else:
                    assert self.dataset == 'TGReDial'
                    for xx in range(int(time_id[ii].cpu())):
                        global_turn_attn.append(xx+1)
                        global_turn_attn.append(xx+1)
                global_sim = F.softmax(torch.tensor(global_turn_attn).cuda().float())
                c_word_true_rep = (global_sim.view(-1, 1) * c_word_attn_rep).sum(dim=0)
                history_words_reps.append(c_word_true_rep)
                
        history_entities_reps = []
        for ii in range(len(h_entities)):
            he = h_entities[ii]
            try:
                yyy=he.item()
                he = entity_indexs[ii].view(1,-1)
            except:
                xxxx=1
            #assert xxxx==1
            c_entity_reps = entity_graph_representations[he]
            c_e_padding_mask = he.eq(self.pad_entity_idx)
            c_entity_attn_rep = self.entity_self_attn(c_entity_reps, c_e_padding_mask)
            simi = F.softmax(torch.matmul(self.entity_weight_matrix(c_entity_attn_rep), entity_attn_rep[ii,:])*10)
            c_entity_true_rep = (simi.view(-1, 1) * c_entity_attn_rep).sum(dim=0)
            history_entities_reps.append(c_entity_true_rep)
            
        history_items_reps = []
        for ii in range(len(h_items)):
            hi = h_items[ii]
            try:
                yyy=hi.item()
                hi = history_items[ii].view(1,-1)
            except:
                xxxx=1
            #assert xxxx==1
            c_item_reps = entity_graph_representations[hi]
            c_i_padding_mask = hi.eq(self.pad_entity_idx)
            c_item_attn_rep = self.entity_self_attn(c_item_reps, c_i_padding_mask)
            simi = F.softmax(torch.matmul(self.entity_weight_matrix(c_item_attn_rep), cur_fusion_rep[ii,:])*10)
            c_item_true_rep = (simi.view(-1, 1) * c_item_attn_rep).sum(dim=0)
            history_items_reps.append(c_item_true_rep)

        history_words_reps = torch.stack(history_words_reps)
        history_entities_reps = torch.stack(history_entities_reps)    
        history_items_reps = torch.stack(history_items_reps)
        
        if get_tr_his_reps:
            return history_words_reps, history_entities_reps, history_items_reps
        
        his_fusion_rep = self.gate_layer(history_entities_reps, history_words_reps)
        
        return history_words_reps, history_entities_reps, history_items_reps, his_fusion_rep, cur_fusion_rep

    def pretrain_infomax(self, batch, neg_batches):
        """
        words: (batch_size, word_length)
        entity_labels: (batch_size, n_entity)
        """
        entity_indexs, words, history_items, history_items_pos, entity_labels, word_labels, item_labels, h_words, h_words_pos, h_entities, h_entities_pos, h_items, user_id, time_id = batch
        neg_entity_indexs_list, neg_words_list, neg_entity_labels_list = neg_batches

        loss_mask_1 = torch.sum(entity_labels)
        if loss_mask_1.item() == 0:
            return None
        loss_mask_2 = torch.sum(word_labels)

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        
        word_representations = word_graph_representations[words]
        word_padding_mask = words.eq(self.pad_word_idx)  # (bs, seq_len)
        
        entity_representations = entity_graph_representations[entity_indexs]
        entity_padding_mask = entity_indexs.eq(self.pad_entity_idx)

        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)

        history_words_reps, history_entities_reps, history_items_reps, his_fusion_rep, _ = self.get_user_historical_info(words, entity_indexs, history_items, h_words, h_words_pos, h_entities, h_items, word_graph_representations, entity_graph_representations, entity_attn_rep, word_attn_rep, time_id)
        
        words_sim_matrix_up = torch.matmul(history_words_reps, history_words_reps.t())
        words_dd = words_sim_matrix_up.diag()
        words_bs = history_words_reps.shape[0]
        words_down1 = torch.sqrt(words_dd.view(-1,1).repeat(1, words_bs))
        words_down2 = torch.sqrt(words_dd.view(1,-1).repeat(words_bs, 1))
        words_sim_matrix = words_sim_matrix_up / words_down1 / words_down2
        words_mask = (words_sim_matrix > 0.85)
        words_sim_matrix = (words_sim_matrix - 0.85 + 0.85*torch.eye(words_bs).cuda()) * words_mask
        
        entities_sim_matrix_up = torch.matmul(history_entities_reps, history_entities_reps.t())
        entities_dd = entities_sim_matrix_up.diag()
        entities_bs = history_entities_reps.shape[0]
        entities_down1 = torch.sqrt(entities_dd.view(-1,1).repeat(1, entities_bs))
        entities_down2 = torch.sqrt(entities_dd.view(1,-1).repeat(entities_bs, 1))
        entities_sim_matrix = entities_sim_matrix_up / entities_down1 / entities_down2
        entities_mask = (entities_sim_matrix > 0.85)
        entities_sim_matrix = (entities_sim_matrix - 0.85 + 0.85*torch.eye(entities_bs).cuda()) * entities_mask
        
        w_user_mask_1 = user_id.view(1,-1).repeat(words_bs, 1)
        w_user_mask_2 = user_id.view(-1,1).repeat(1, words_bs)
        user_mask = (w_user_mask_1!=w_user_mask_2) + torch.eye(words_bs).cuda()
        words_sim_matrix = words_sim_matrix * user_mask
        
        e_user_mask_1 = user_id.view(1,-1).repeat(entities_bs, 1)
        e_user_mask_2 = user_id.view(-1,1).repeat(1, entities_bs)
        user_mask = (e_user_mask_1!=e_user_mask_2) + torch.eye(entities_bs).cuda()
        entities_sim_matrix = entities_sim_matrix * user_mask
        
        word_attn_rep_weighted = []
        for i in range(words_bs):
            word_attn_rep_weighted.append((word_attn_rep * words_sim_matrix[i].view(-1, 1)).sum(dim=0))
            
        entity_attn_rep_weighted = []
        for i in range(entities_bs):
            entity_attn_rep_weighted.append((entity_attn_rep * entities_sim_matrix[i].view(-1, 1)).sum(dim=0))
        word_attn_rep_add_lookalike = torch.stack(word_attn_rep_weighted)
        entity_attn_rep_add_lookalike = torch.stack(entity_attn_rep_weighted)
        
        
        word_info_rep = self.word_infomax_norm(word_attn_rep_add_lookalike)
        entity_info_rep = self.entity_infomax_norm(entity_attn_rep_add_lookalike)
        history_word_rep = self.word_infomax_norm(history_words_reps)
        history_entity_rep = self.entity_infomax_norm(history_entities_reps)
        
        
        his_part1, his_part2, loss_1 = self.compute_barrow_loss(history_word_rep, history_entity_rep, 0.1)
        cur_part1, cur_part2, loss_2 = self.compute_barrow_loss(word_info_rep, entity_info_rep, 0.1)
        cross_part1, cross_part2, loss_3 = self.compute_barrow_loss(his_fusion_rep, history_items_reps, 0.1)
        
        loss = loss_1 + loss_2 + loss_3
        #print(pred_item_loss)
        return None, loss

    def get_tr_history_reps(self, batch, neg_batches, mode):
        context_entities, context_words, history_items, history_items_pos, entities, word_labels, item_labels, h_words, h_words_pos, h_entities, h_entities_pos, h_items, user_id, time_id, movie = batch
        neg_entity_indexs_list, neg_words_list, neg_entity_labels_list = neg_batches

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]
        
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        
        
        history_words_reps, history_entities_reps, history_items_reps = self.get_user_historical_info(context_words, context_entities, history_items, h_words, h_words_pos, h_entities, h_items, word_graph_representations, entity_graph_representations, entity_attn_rep, word_attn_rep, time_id, get_tr_his_reps=True)

        history_words_reps = history_words_reps.detach().cpu()
        history_entities_reps = history_entities_reps.detach().cpu()
        history_items_reps = history_items_reps.detach().cpu()
        
        entity_attn_rep = entity_attn_rep.detach().cpu()
        word_attn_rep = word_attn_rep.detach().cpu()
        
        return history_words_reps.cpu(), history_entities_reps.cpu(), history_items_reps.cpu(), word_attn_rep.cpu(), entity_attn_rep.cpu(), user_id.cpu()
    
    
    def get_final_rep_train_test(self, tr_his_words_reps, history_words_reps, tr_his_entities_reps, history_entities_reps, tr_user_id, user_id, word_attn_rep, tr_word_attn_rep, entity_attn_rep, tr_entity_attn_rep, history_items_reps, user_rep_cur, tau_e=5, delta_e=0.85):
        
        words_sim_matrix_up = torch.matmul(tr_his_words_reps.detach(), history_words_reps.t())
        words_bs_tr = tr_his_words_reps.shape[0]
        words_bs_te = history_words_reps.shape[0]
        words_dd1 = torch.matmul(tr_his_words_reps.detach(), tr_his_words_reps.detach().t()).diag()
        words_dd2 = torch.matmul(history_words_reps, history_words_reps.t()).diag()
        words_down1 = torch.sqrt(words_dd1.view(-1,1).repeat(1, words_bs_te))
        words_down2 = torch.sqrt(words_dd2.view(1,-1).repeat(words_bs_tr, 1))
        words_sim_matrix = words_sim_matrix_up / words_down1 / words_down2
        words_mask = (words_sim_matrix > 0.85)
        words_sim_matrix = (words_sim_matrix - 0.85) * words_mask
        
        entities_sim_matrix_up = torch.matmul(tr_his_entities_reps.detach(), history_entities_reps.t())
        entities_bs_tr = tr_his_entities_reps.shape[0]
        entities_bs_te = history_entities_reps.shape[0]
        entities_dd1 = torch.matmul(tr_his_entities_reps.detach(), tr_his_entities_reps.detach().t()).diag()
        entities_dd2 = torch.matmul(history_entities_reps, history_entities_reps.t()).diag()
        entities_down1 = torch.sqrt(entities_dd1.view(-1,1).repeat(1, entities_bs_te))
        entities_down2 = torch.sqrt(entities_dd2.view(1,-1).repeat(entities_bs_tr, 1))
        entities_sim_matrix = entities_sim_matrix_up / entities_down1 / entities_down2
        entities_mask = (entities_sim_matrix > delta_e)
        entities_sim_matrix = (entities_sim_matrix - delta_e) * entities_mask       
        
        w_user_mask_1 = tr_user_id.cuda().view(-1,1).repeat(1, words_bs_te)
        w_user_mask_2 = user_id.view(1,-1).repeat(words_bs_tr, 1)
        user_mask = (w_user_mask_1!=w_user_mask_2)
        words_sim_matrix = words_sim_matrix * user_mask
        
        e_user_mask_1 = tr_user_id.cuda().view(-1,1).repeat(1, entities_bs_te)
        e_user_mask_2 = user_id.view(1,-1).repeat(entities_bs_tr, 1)
        user_mask = (e_user_mask_1!=e_user_mask_2)
        entities_sim_matrix = entities_sim_matrix * user_mask
        
        word_attn_rep_weighted = []
        for i in range(words_bs_te):
            word_attn_rep_weighted.append(word_attn_rep[i] + (tr_word_attn_rep.detach() * words_sim_matrix[:,i].view(-1, 1)).sum(dim=0))
            
        entity_attn_rep_weighted = []
        for i in range(entities_bs_te):
            entity_attn_rep_weighted.append(entity_attn_rep[i] + (tr_entity_attn_rep.detach() * entities_sim_matrix[:,i].view(-1, 1)).sum(dim=0))
        
        word_attn_rep_add_lookalike = torch.stack(word_attn_rep_weighted)
        entity_attn_rep_add_lookalike = torch.stack(entity_attn_rep_weighted)
        
        his_ff_user_rep = self.gate_layer(history_words_reps.cuda(), history_entities_reps.cuda())
        
        sim_past_cur_up = torch.matmul(his_ff_user_rep, user_rep_cur.t()).diag()
        sim_past_cur_down1 = torch.matmul(his_ff_user_rep, his_ff_user_rep.t()).diag()
        sim_past_cur_down2 = torch.matmul(user_rep_cur, user_rep_cur.t()).diag()
        sim_past_cur = sim_past_cur_up / torch.sqrt(sim_past_cur_down1) / torch.sqrt(sim_past_cur_down2)
        sim_past_cur = sim_past_cur.view(-1, 1)
        sim_past_cur_mask = (sim_past_cur>0.85)
        sim_past_cur = (sim_past_cur - 0.85) * sim_past_cur_mask
        
        fff1 = torch.cat([history_words_reps, word_attn_rep], dim=1)
        past_cur_sim1 = nn.Sigmoid()(self.past_cur_rep_sigma(fff1)) / tau_e
        fff2 = torch.cat([history_entities_reps, entity_attn_rep], dim=1)
        past_cur_sim2 = nn.Sigmoid()(self.past_cur_rep_sigma(fff2)) / tau_e
        past_cur_sim = (past_cur_sim1 + past_cur_sim2) / 2
        
        entity_attn_rep_final = entity_attn_rep_add_lookalike + past_cur_sim2 * history_entities_reps
        word_attn_rep_final = word_attn_rep_add_lookalike + past_cur_sim1 * history_words_reps
        user_rep_final = self.gate_layer2(entity_attn_rep_final, word_attn_rep_final)
        user_rep = user_rep_final + sim_past_cur.view(-1, 1) * history_items_reps
        user_rep = user_rep.cuda()
        
        return user_rep, entity_attn_rep_add_lookalike, word_attn_rep_add_lookalike, history_words_reps, history_entities_reps
    
    def recommend_test(self, batch, neg_batches, tr_his_words_reps, tr_his_entities_reps, tr_his_items_reps, tr_word_attn_rep, tr_entity_attn_rep, tr_user_id, mode):
        delta_e = 0.85
        tau_e = 6
        
        context_entities, context_words, history_items, history_items_pos, entities, word_labels, item_labels, h_words, h_words_pos, h_entities, h_entities_pos, h_items, user_id, time_id, movie = batch
        neg_entity_indexs_list, neg_words_list, neg_entity_labels_list = neg_batches

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]
        
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        
        history_words_reps, history_entities_reps, history_items_reps, _, user_rep_cur = self.get_user_historical_info(context_words, context_entities, history_items, h_words, h_words_pos, h_entities, h_items, word_graph_representations, entity_graph_representations, entity_attn_rep, word_attn_rep, time_id)
        
        user_rep, _, _, _ , _ = self.get_final_rep_train_test(tr_his_words_reps, history_words_reps, tr_his_entities_reps, history_entities_reps, tr_user_id, user_id, word_attn_rep, tr_word_attn_rep, entity_attn_rep, tr_entity_attn_rep, history_items_reps, user_rep_cur)
        
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, movie)
        
        
        return rec_loss, None, rec_scores
    
    
    def recommend_training(self, batch, neg_batches, tr_his_words_reps, tr_his_entities_reps, tr_his_items_reps, tr_word_attn_rep, tr_entity_attn_rep, tr_user_id, mode):
        delta_e = 0.85
        tau_e = 6
        
        context_entities, context_words, history_items, history_items_pos, entities, word_labels, item_labels, h_words, h_words_pos, h_entities, h_entities_pos, h_items, user_id, time_id, movie = batch
        neg_entity_indexs_list, neg_words_list, neg_entity_labels_list = neg_batches

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, word_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]
        
        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)
        
        history_words_reps, history_entities_reps, history_items_reps, _, user_rep_cur = self.get_user_historical_info(context_words, context_entities, history_items, h_words, h_words_pos, h_entities, h_items, word_graph_representations, entity_graph_representations, entity_attn_rep, word_attn_rep, time_id)
        
        user_rep, entity_attn_rep_add_lookalike, word_attn_rep_add_lookalike, history_words_reps, history_entities_reps = self.get_final_rep_train_test(tr_his_words_reps, history_words_reps, tr_his_entities_reps, history_entities_reps, tr_user_id, user_id, word_attn_rep, tr_word_attn_rep, entity_attn_rep, tr_entity_attn_rep, history_items_reps, user_rep_cur)
        
        rec_scores = F.linear(user_rep, entity_graph_representations, self.rec_bias.bias)  # (bs, #entity)

        rec_loss = self.rec_loss(rec_scores, movie)

        info_loss_mask_1 = torch.sum(entities)
        info_loss_mask_2 = torch.sum(word_labels)
        info_loss_mask_3 = torch.sum(entities)
        if info_loss_mask_1.item() == 0:
            info_loss = None
        else:
            
            word_info_rep = self.word_infomax_norm(word_attn_rep_add_lookalike)  # (bs, dim)
            entity_info_rep = self.entity_infomax_norm(entity_attn_rep_add_lookalike)
            history_word_rep = self.word_infomax_norm(history_words_reps)
            history_entity_rep = self.entity_infomax_norm(history_entities_reps)
            his_fusion_rep = self.gate_layer(history_entities_reps, history_words_reps)
            
            his_part1, his_part2, info_loss_1 = self.compute_barrow_loss(history_word_rep, history_entity_rep, 0.1)
            cur_part1, cur_part2, info_loss_2 = self.compute_barrow_loss(word_info_rep, entity_info_rep, 0.1)
            cross_part1, cross_part2, info_loss_3 = self.compute_barrow_loss(his_fusion_rep, history_items_reps, 0.1)
            
            info_loss = info_loss_1 + info_loss_2 + info_loss_3

        return rec_loss, info_loss, rec_scores
        

    def freeze_parameters(self):
        freeze_models = [self.word_kg_embedding, self.entity_encoder, self.entity_self_attn, self.word_encoder,
                         self.word_self_attn, self.gate_layer, self.infomax_bias, self.word_infomax_norm, self.entity_infomax_norm, self.rec_bias]
        for model in freeze_models:
            for p in model.parameters():
                p.requires_grad = False

    def _starts(self, batch_size):
        """Return bsz start tokens."""
        return self.START.detach().expand(batch_size, 1)

    def _decode_forced_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, response):
        batch_size, seq_len = response.shape
        start = self._starts(batch_size)
        inputs = torch.cat((start, response[:, :-1]), dim=-1).long()

        dialog_latent, _ = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                             entity_reps, entity_mask)  # (bs, seq_len, dim)
        entity_latent = entity_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        word_latent = word_emb_attn.unsqueeze(1).expand(-1, seq_len, -1)
        copy_latent = self.copy_norm(
            torch.cat((entity_latent, word_latent, dialog_latent), dim=-1))  # (bs, seq_len, dim)

        copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(
            0)  # (bs, seq_len, vocab_size)
        gen_logits = F.linear(dialog_latent, self.token_embedding.weight)  # (bs, seq_len, vocab_size)
        sum_logits = copy_logits + gen_logits
        preds = sum_logits.argmax(dim=-1)
        return sum_logits, preds

    def _decode_greedy_with_kg(self, token_encoding, entity_reps, entity_emb_attn, entity_mask,
                               word_reps, word_emb_attn, word_mask, his_user_rep):
        batch_size = token_encoding[0].shape[0]
        inputs = self._starts(batch_size).long()
        incr_state = None
        logits = []
        for _ in range(self.response_truncate):
            dialog_latent, incr_state = self.conv_decoder(inputs, token_encoding, word_reps, word_mask,
                                                          entity_reps, entity_mask, incr_state)
            dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
            db_latent = entity_emb_attn.unsqueeze(1)
            concept_latent = word_emb_attn.unsqueeze(1)
            copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

            copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
            gen_logits = F.linear(dialog_latent, self.token_embedding.weight)
            
            sum_logits = copy_logits + gen_logits
            
            seq_lenn = sum_logits.shape[1]
            user_vocab_bias = self.user_rep_to_vocab(his_user_rep)
            user_vocab_bias = user_vocab_bias.unsqueeze(1).repeat(1,seq_lenn,1)
            sum_logits = sum_logits + user_vocab_bias
            
            preds = sum_logits.argmax(dim=-1).long()
            logits.append(sum_logits)
            inputs = torch.cat((inputs, preds), dim=1)

            finished = ((inputs == self.end_token_idx).sum(dim=-1) > 0).sum().item() == batch_size
            if finished:
                break
        logits = torch.cat(logits, dim=1)
        return logits, inputs
    
    def converse(self, batch, mode):
        
        #context_entities, context_words, history_items, history_items_pos, entities, word_labels, item_labels, h_words, h_entities, h_items, user_id, time_id, movie
        
        context_tokens, context_entities, context_words, history_items, h_words, h_entities, h_items, user_id, response = batch

        entity_graph_representations = self.entity_encoder(None, self.entity_edge_idx, self.entity_edge_type)
        word_graph_representations = self.word_encoder(self.word_kg_embedding.weight, self.word_edges)

        entity_padding_mask = context_entities.eq(self.pad_entity_idx)  # (bs, entity_len)
        word_padding_mask = context_words.eq(self.pad_word_idx)  # (bs, seq_len)

        entity_representations = entity_graph_representations[context_entities]
        word_representations = word_graph_representations[context_words]

        entity_attn_rep = self.entity_self_attn(entity_representations, entity_padding_mask)
        word_attn_rep = self.word_self_attn(word_representations, word_padding_mask)

         
        user_rep = self.gate_layer(entity_attn_rep, word_attn_rep)
        
        history_words_reps = []
        for ii in range(len(h_words)):
            hw = h_words[ii]
            try:
                yyy=hw.item()
                hw = context_words[ii].view(1,-1)
            except:
                xxxx=1
            #assert xxxx==1
            if self.dataset == 'ReDial' and int((hw==0).sum().cpu().numpy())==hw.shape[1]:
                c_word_reps = entity_graph_representations[hw]
                c_w_padding_mask = hw.eq(self.pad_entity_idx)
            else:
                c_word_reps = word_graph_representations[hw]
                c_w_padding_mask = hw.eq(self.pad_word_idx)
            c_word_attn_rep = self.word_self_attn(c_word_reps, c_w_padding_mask)
            simi = F.softmax(torch.matmul(self.word_weight_matrix(c_word_attn_rep), word_attn_rep[ii,:])*10)
            c_word_true_rep = (simi.view(-1, 1) * c_word_attn_rep).sum(dim=0)
            history_words_reps.append(c_word_true_rep)
            
        history_entities_reps = []
        for ii in range(len(h_entities)):
            he = h_entities[ii]
            try:
                yyy=he.item()
                he = context_entities[ii].view(1,-1)
            except:
                xxxx=1
            #assert xxxx==1
            c_entity_reps = entity_graph_representations[he]
            c_e_padding_mask = he.eq(self.pad_entity_idx)
            c_entity_attn_rep = self.entity_self_attn(c_entity_reps, c_e_padding_mask)
            simi = F.softmax(torch.matmul(self.entity_weight_matrix(c_entity_attn_rep), entity_attn_rep[ii,:])*10)
            c_entity_true_rep = (simi.view(-1, 1) * c_entity_attn_rep).sum(dim=0)
            history_entities_reps.append(c_entity_true_rep)

        history_items_reps = []
        for ii in range(len(h_items)):
            hi = h_items[ii]
            try:
                yyy=hi.item()
                hi = history_items[ii].view(1,-1)
            except:
                xxxx=1
            #assert xxxx==1
            c_item_reps = entity_graph_representations[hi]
            c_i_padding_mask = hi.eq(self.pad_entity_idx)
            c_item_attn_rep = self.entity_self_attn(c_item_reps, c_i_padding_mask)
            simi = F.softmax(torch.matmul(self.entity_weight_matrix(c_item_attn_rep), user_rep[ii,:])*10)
            c_item_true_rep = (simi.view(-1, 1) * c_item_attn_rep).sum(dim=0)
            history_items_reps.append(c_item_true_rep)

        history_words_reps = torch.stack(history_words_reps)
        history_entities_reps = torch.stack(history_entities_reps)
        history_items_reps = torch.stack(history_items_reps)
        

        words_sim_matrix_up = torch.matmul(history_words_reps, history_words_reps.t())
        words_dd = words_sim_matrix_up.diag()
        words_bs = history_words_reps.shape[0]
        words_down1 = torch.sqrt(words_dd.view(-1,1).repeat(1, words_bs))
        words_down2 = torch.sqrt(words_dd.view(1,-1).repeat(words_bs, 1))
        words_sim_matrix = words_sim_matrix_up / words_down1 / words_down2
        words_mask = (words_sim_matrix > 0.85)
        words_sim_matrix = (words_sim_matrix - 0.85 + 0.85*torch.eye(words_bs).cuda()) * words_mask
        
        entities_sim_matrix_up = torch.matmul(history_entities_reps, history_entities_reps.t())
        entities_dd = entities_sim_matrix_up.diag()
        entities_bs = history_entities_reps.shape[0]
        entities_down1 = torch.sqrt(entities_dd.view(-1,1).repeat(1, entities_bs))
        entities_down2 = torch.sqrt(entities_dd.view(1,-1).repeat(entities_bs, 1))
        entities_sim_matrix = entities_sim_matrix_up / entities_down1 / entities_down2
        entities_mask = (entities_sim_matrix > 0.85)
        entities_sim_matrix = (entities_sim_matrix - 0.85 + 0.85*torch.eye(entities_bs).cuda()) * entities_mask
        
        w_user_mask_1 = user_id.view(1,-1).repeat(words_bs, 1)
        w_user_mask_2 = user_id.view(-1,1).repeat(1, words_bs)
        user_mask = (w_user_mask_1!=w_user_mask_2) + torch.eye(words_bs).cuda()
        words_sim_matrix = words_sim_matrix * user_mask
        
        e_user_mask_1 = user_id.view(1,-1).repeat(entities_bs, 1)
        e_user_mask_2 = user_id.view(-1,1).repeat(1, entities_bs)
        user_mask = (e_user_mask_1!=e_user_mask_2) + torch.eye(entities_bs).cuda()
        entities_sim_matrix = entities_sim_matrix * user_mask
        
        word_attn_rep_weighted = []
        for i in range(words_bs):
            word_attn_rep_weighted.append((word_attn_rep * words_sim_matrix[i].view(-1, 1)).sum(dim=0))
            
        entity_attn_rep_weighted = []
        for i in range(entities_bs):
            entity_attn_rep_weighted.append((entity_attn_rep * entities_sim_matrix[i].view(-1, 1)).sum(dim=0))
        word_attn_rep_wei = torch.stack(word_attn_rep_weighted)
        entity_attn_rep_wei = torch.stack(entity_attn_rep_weighted)
        
        his_ff_user_rep = self.gate_layer(history_words_reps, history_entities_reps)
        sim_past_cur_up = torch.matmul(his_ff_user_rep, user_rep.t()).diag()
        sim_past_cur_down1 = torch.matmul(his_ff_user_rep, his_ff_user_rep.t()).diag()
        sim_past_cur_down2 = torch.matmul(user_rep, user_rep.t()).diag()
        sim_past_cur = sim_past_cur_up / torch.sqrt(sim_past_cur_down1) / torch.sqrt(sim_past_cur_down2)
        sim_past_cur = sim_past_cur.view(-1, 1)
        sim_past_cur_mask = (sim_past_cur>0.85)
        sim_past_cur = (sim_past_cur - 0.85) * sim_past_cur_mask
        user_rep = self.gate_layer(entity_attn_rep_wei, word_attn_rep_wei) + sim_past_cur.view(-1, 1) * history_items_reps
        
        # encoder-decoder
        tokens_encoding = self.conv_encoder(context_tokens)
        conv_entity_emb = self.conv_entity_attn_norm(entity_attn_rep)
        conv_word_emb = self.conv_word_attn_norm(word_attn_rep)
        conv_entity_reps = self.conv_entity_norm(entity_representations)
        conv_word_reps = self.conv_word_norm(word_representations)

        if mode != 'test':
            logits, preds = self._decode_forced_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,
                                                        response)

            seq_lenn = logits.shape[1]
            user_vocab_bias = self.user_rep_to_vocab(user_rep)
            user_vocab_bias = user_vocab_bias.unsqueeze(1).repeat(1,seq_lenn,1)
            logits = logits + user_vocab_bias
            logits = logits.view(-1, logits.shape[-1])
            response = response.view(-1)
            loss = self.conv_loss(logits, response)
            return loss, preds
        else:
            logits, preds = self._decode_greedy_with_kg(tokens_encoding, conv_entity_reps, conv_entity_emb,
                                                        entity_padding_mask,
                                                        conv_word_reps, conv_word_emb, word_padding_mask,user_rep)
            return preds
