import os
import numpy as np


class SEMLoad(object):
    def __init__(self, root_path, train=True, max_len=100, limit=50):
        self.root_path = root_path
        self.train = train
        self.max_len = max_len
        self.limit = limit

        if self.train:
            print("Loading the training set ...")
            self.train_path = os.path.join(root_path, "train.txt")
        else:
            print("Loading the test set ...")
            self.test_path = os.path.join(root_path, "test.txt")

        self.rel_path = os.path.join(root_path, "relation2id.txt")
        self.w2v_path = os.path.join(root_path, "vector_50.txt")

        print("Loading Start ...")
        # loading from relation2id.txt
        self.rel2id, self.id2rel = self.load_rel()
        # loading from vector_50.txt
        self.w2v, self.word2id, self.id2word = self.load_w2v()
        if train:

            self.lexical_feature, sen_feature, self.labels = self.parsing(self.train_path)
        else:
            self.lexical_feature, sen_feature, self.labels = self.parsing(self.test_path)

        self.word_feautre, self.left_pf, self.right_pf = sen_feature

        print('loading finish')

    def save(self):
        if self.train:
            prefix = 'train'
        else:
            prefix = 'test'
        np.save(os.path.join(self.root_path, prefix, 'word_feature.npy'), self.word_feautre)
        np.save(os.path.join(self.root_path, prefix, 'left_pf.npy'), self.left_pf)
        np.save(os.path.join(self.root_path, prefix, 'right_pf.npy'), self.right_pf)
        np.save(os.path.join(self.root_path, prefix, 'lexical_feature.npy'), self.lexical_feature)
        np.save(os.path.join(self.root_path, prefix, 'labels.npy'), self.labels)
        np.save(os.path.join(self.root_path, prefix, 'w2v.npy'), self.w2v)
        print('save finish!')

    def load_rel(self):
        relation = [line.strip('\n').split() for line in open(self.rel_path)]
        rel2id = {int(i): j for i, j in relation}
        id2rel = {i: int(j) for j, i in relation}
        return rel2id, id2rel

    def load_w2v(self):
        word_list = []
        vec_list = []

        with open(self.w2v_path) as w2v:
            for line in w2v:
                line = line.strip('\n').split()
                word = line[0]
                vector = list(map(float, line[1:]))
                word_list.append(word)
                vec_list.append(vector)

        word2id = {word: index for index, word in enumerate(word_list)}
        id2word = {index: word for index, word in enumerate(word_list)}

        return np.array(vec_list, dtype=np.float32), word2id, id2word

    def parsing(self, path):
        input_info = []
        label_info = []

        with open(path) as lines:
            for line in lines:
                line = line.strip('\n').split(' ')
                rel_tmp = int(line[0])
                ent1 = (int(line[1]), int(line[2]))
                ent2 = (int(line[3]), int(line[4]))

                sen_input = line[5:]
                token_input = list(map(lambda x: self.word2id.get(x, self.word2id['<PAD>']), sen_input))
                label_info.append(rel_tmp)
                input_info.append((ent1, ent2, token_input))

        lexical_feature = self.construct_lexical_feature(input_info)
        sentence_feature = self.contruct_sentence_feature(input_info)

        return lexical_feature, sentence_feature, label_info

    def construct_lexical_feature(self, input_info):
        lexical_feature = []
        for input in input_info:
            ent1, ent2, tokens = input
            left_e1 = self.get_left(ent1, tokens)
            left_e2 = self.get_left(ent2, tokens)
            right_e1 = self.get_right(ent1, tokens)
            right_e2 = self.get_right(ent2, tokens)
            lexical_feature.append((left_e1, tokens[ent1[0]], right_e1, left_e2, tokens[ent2[0]], right_e2))

        return lexical_feature

    def contruct_sentence_feature(self, input_info):
        """
        :param input_info: a list of input with format ((ent1-begin,ent1-end),  (ent2-begin,ent2-end), tokens)
        :return: tokens sequence, pos_to_e1 sequence, pos_to_e2 sequence.

        """
        update_sen = []

        for input in input_info:
            ent1, ent2, tokens = input
            pos_to_e1 = []
            pos_to_e2 = []

            for index in range(len(tokens)):
                pos_to_e1.append(self.get_pos_feature(index - ent1[0]))
                pos_to_e2.append(self.get_pos_feature(index - ent2[0]))
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
                pos_to_e1 = pos_to_e1[:self.max_len]
                pos_to_e2 = pos_to_e2[:self.max_len]
            elif len(tokens) < self.max_len:
                pos_to_e1.extend([self.limit * 2 + 2] * (self.max_len - len(tokens)))
                pos_to_e2.extend([self.limit * 2 + 2] * (self.max_len - len(tokens)))
                tokens.extend([self.word2id['<PAD>']] * (self.max_len - len(tokens)))
            update_sen.append([tokens, pos_to_e1, pos_to_e2])

        return zip(*update_sen)

    def get_left(self, entity, tokens):
        """
        entity : [begin_idx, end_idx]
        tokens : tokens sequence of the input.

        Return : the index of the adjacent word to the left of the entity
        """
        pos = entity[0]
        if pos > 0:
            return tokens[pos-1]
        else:
            return self.word2id['<PAD>']

    def get_right(self, entity, tokens):
        """
                entity : [begin_idx, end_idx]
                tokens : tokens sequence of the input.

                Return : the index of the adjacent word to the right of the entity
        """
        pos = entity[1]
        if pos < len(tokens) - 1:
            return tokens[pos + 1]
        else:
            return self.word2id['<PAD>']


    def get_pos_feature(self, x):
        '''
        map the relative position into [0, 2 * limit + 1]
        '''
        if x < -self.limit:
            return 0
        if -self.limit <= x <= self.limit:
            return x + self.limit + 1
        if x > self.limit:
            return self.limit * 2 + 1


if __name__ == "__main__":
    # processing & save the train set
    DataLoader = SEMLoad("./sem-task8", train=True)
    DataLoader.save()
    # processing & save the test set
    TestLoader = SEMLoad("./sem-task8", train=False)
    TestLoader.save()






