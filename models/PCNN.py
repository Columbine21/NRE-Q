import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModel import BasicModel


class PCNN(BasicModel):
    """
        the basic model
        Zeng 2014 "Relation Classification via Convolutional Deep Neural Network"
    """
    def __init__(self, opt):
        super(PCNN, self).__init__()

        self.opt = opt
        self.model_name = opt.model

        self.word_embedding = nn.Embedding(self.opt.vocab_size, self.opt.word_dim)
        self.pos1_embedding = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)
        self.pos2_embedding = nn.Embedding(self.opt.pos_size + 1, self.opt.pos_dim)

        feature_dim = self.opt.word_dim + 2 * self.opt.pos_dim

        self.convs = nn.ModuleList([nn.Conv2d(1, self.opt.filters_num, (k, feature_dim), padding=(int(k / 2), 0))
                                    for k in self.opt.filters])
        # In our model self.opt.filters = {3}
        self.cnn_linear = nn.Linear(len(self.opt.filters) * self.opt.filters_num, self.opt.sen_feature_dim)

        self.out_linear = nn.Linear(len(self.opt.filters) * self.opt.filters_num + 6 * self.opt.word_dim, self.opt.rel_num)

        self.dropout = nn.Dropout(self.opt.drop_out)

        # init word embedding & model.parameters().
        self.init_model_weight()

        self.init_word_emb()

    def init_word_emb(self):

        w2v = torch.from_numpy(np.load(self.opt.w2v_path))

        self.word_embedding.weight.data.copy_(w2v)

    def init_model_weight(self):
        for param in self.parameters():
            # print(param)
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
            else:
                nn.init.constant_(param, 0.)

    def forward(self, input):
        word_features, left_pf, right_pf, lexical_features = input

        batch_size = lexical_features.size(0)

        lexical_level_emb = self.word_embedding(lexical_features)
        # print(lexical_level_emb.shape)
        assert lexical_level_emb.shape == torch.Size([batch_size, 6, self.opt.word_dim]), \
            "Lexical Feature shape incorrect."
        lexical_level_emb = lexical_level_emb.view(batch_size, -1)

        word_emb = self.word_embedding(word_features)
        assert word_emb.shape == torch.Size([batch_size, 100, self.opt.word_dim])
        pos1_emb = self.pos1_embedding(left_pf)
        assert pos1_emb.shape == torch.Size([batch_size, 100, self.opt.pos_dim])
        pos2_emb = self.pos2_embedding(right_pf)
        assert pos2_emb.shape == torch.Size([batch_size, 100, self.opt.pos_dim])

        sentence_feature = torch.cat([word_emb, pos1_emb, pos2_emb], dim=2)
        assert sentence_feature.shape == torch.Size([batch_size, 100, self.opt.word_dim+2*self.opt.pos_dim])

        # conv part
        sentence_feature = sentence_feature.unsqueeze(1)
        # sentence_feature = [batch_size, 1, sen_len, self.opt.word_dim+2*self.opt.pos_dim]
        sentence_feature = self.dropout(sentence_feature)
        sentence_feature = [F.relu(conv(sentence_feature)).squeeze(3) for conv in self.convs]
        sentence_feature = [F.max_pool1d(sentence_feature_tmp, sentence_feature_tmp.size(2)).squeeze(2)
                            for sentence_feature_tmp in sentence_feature]
        sentence_feature = torch.cat(sentence_feature, 1)

        sentence_level_emb = sentence_feature

        feature = torch.cat([lexical_level_emb, sentence_level_emb], dim=1)
        feature = self.dropout(feature)

        clf_result = self.out_linear(feature)

        return clf_result





