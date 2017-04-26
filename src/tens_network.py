"""
Rewrite network by tensflow
"""

from __future__ import print_function
from __future__ import division

import time
import random
import sys

import tensorflow as tf
import numpy as np

import tensorflow.contrib.rnn as rnn
from phrase_tree import PhraseTree, FScore
from features import FeatureMapper
from parser import Parser

def wrapper_indices(b):
    c = tf.expand_dims(b, -1)
    d = tf.stack([tf.range(0, tf.shape(b)[0], dtype=tf.int32) for i in range(b.get_shape()[1])])
    e = tf.matrix_transpose(d)
    f = tf.expand_dims(e, -1)
    g = tf.concat([f,c], axis=-1)
    return g

def gather_axis(params, indices):
    wrap = wrapper_indices(indices)
    return tf.gather_nd(params, wrap)



def another_gather(params, indices):
    c = tf.expand_dims(indices, -1)
    d = tf.range(0, tf.shape(indices)[0], dtype=tf.int32)
    e = tf.expand_dims(d, -1)
    f = tf.concat([e,c], axis=-1)
    return tf.gather_nd(params, f)

def prapare_data(data_list, max_len):
    ans = {}
    ans_struct_word = []
    ans_struct_tag = []
    ans_struct_seq_len = []
    ans_struct_lefts = []
    ans_struct_rights = []
    ans_struct_actions = []
    ans_label_word = []
    ans_label_tag = []
    ans_label_seq_len = []
    ans_label_lefts = []
    ans_label_rights = []
    ans_label_actions = []
    for data in data_list:
        len_of_seq = len(data['w'])
        padding_word = data['w'].tolist()
        for i in xrange(max_len-len_of_seq):
            padding_word.append(0)
        padding_tag = data['t'].tolist()
        for i in xrange(max_len-len_of_seq):
            padding_tag.append(0)
        for features, actions in data['struct_data'].items():
            ans_struct_seq_len.append(len_of_seq)
            ans_struct_word.append(padding_word)
            ans_struct_tag.append(padding_tag)
            ans_struct_lefts.append(features[0])
            ans_struct_rights.append(features[1])
            ans_struct_actions.append(actions)
        for features, actions in data['label_data'].items():
            ans_label_seq_len.append(len_of_seq)
            ans_label_word.append(padding_word)
            ans_label_tag.append(padding_tag)
            ans_label_lefts.append(features[0])
            ans_label_rights.append(features[1])
            ans_label_actions.append(actions)
    ans['struct_word'] = np.array(ans_struct_word)
    ans['struct_tag'] = np.array(ans_struct_tag)
    ans['struct_seq_len'] = np.array(ans_struct_seq_len)
    ans['label_word'] = np.array(ans_label_word)
    ans['label_tag'] = np.array(ans_label_tag)
    ans['label_seq_len'] = np.array(ans_label_seq_len)
    ans['struct_lefts'] = np.array(ans_struct_lefts)
    ans['struct_rights'] = np.array(ans_struct_rights)
    ans['struct_actions'] = np.array(ans_struct_actions)
    ans['label_lefts'] = np.array(ans_label_lefts)
    ans['label_rights'] = np.array(ans_label_rights)
    ans['label_actions'] = np.array(ans_label_actions)
    return ans

class Network(object):

    def __init__(
        self,
        word_count,
        tag_count,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        struct_out,
        label_out,
        struct_spans=4,
        label_spans=3,
        max_seq_len=150
    ):

        self.word_count = word_count
        self.tag_count = tag_count
        self.word_dims = word_dims
        self.tag_dims = tag_dims
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.struct_out = struct_out
        self.label_out = label_out
        self.max_seq_len = max_seq_len
        self.struct_spans = struct_spans
        self.label_spans = label_spans
        random.seed(1)

        self.word_embed = tf.Variable(tf.random_uniform(
            (word_count, word_dims), -0.01, 0.01, dtype=tf.float32), name='word_embed', dtype=tf.float32)
        self.tag_embed = tf.Variable(tf.random_uniform(
            (tag_count, tag_dims), -0.01, 0.01, dtype=tf.float32), name='tag_embed', dtype=tf.float32)

        self.struct_word_inds = tf.placeholder(
            tf.int32, [None, self.max_seq_len], name='struct_word_inds')
        self.struct_tag_inds = tf.placeholder(
            tf.int32, [None, self.max_seq_len], name='struct_tag_inds')
        self.struct_seq_len = tf.placeholder(
            tf.int32, [None], name='struct_seq_len')

        self.label_word_inds = tf.placeholder(
            tf.int32, [None, self.max_seq_len], name='label_word_inds')
        self.label_tag_inds = tf.placeholder(
            tf.int32, [None, self.max_seq_len], name='label_tag_inds')
        self.label_seq_len = tf.placeholder(
            tf.int32, [None], name='label_seq_len')

        self.struct_lefts = tf.placeholder(
            tf.int32, [None, struct_spans], name='struct_lefts')
        self.label_lefts = tf.placeholder(
            tf.int32, [None, label_spans], name='label_lefts')
        self.struct_rights = tf.placeholder(
            tf.int32, [None, struct_spans], name='struct_rights')
        self.label_rights = tf.placeholder(
            tf.int32, [None, label_spans], name='label_rights')
        self.struct_actions = tf.placeholder(
            tf.int32, [None], name='struct_actions')
        self.label_actions = tf.placeholder(
            tf.int32, [None], name='label_actions')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.fwd1 = rnn.LSTMCell(self.lstm_units, forget_bias=1.0)
        self.back1 = rnn.LSTMCell(self.lstm_units, forget_bias=1.0)

        self.fwd2 = rnn.LSTMCell(self.lstm_units, forget_bias=1.0)
        self.back2 = rnn.LSTMCell(self.lstm_units, forget_bias=1.0)

        with tf.variable_scope('rnn') as scope:
            self.struct_fwd_lstm_layer, self.struct_back_lstm_layer = self.evaluate_recurrent(
                self.struct_word_inds, self.struct_tag_inds, self.struct_seq_len, self.keep_prob)
            scope.reuse_variables()
            self.label_fwd_lstm_layer, self.label_back_lstm_layer = self.evaluate_recurrent(
                self.label_word_inds, self.label_tag_inds, self.label_seq_len, self.keep_prob)
        self.struct_feature_layer = self.gather_lstm_feature(
            self.struct_fwd_lstm_layer, self.struct_back_lstm_layer, self.struct_lefts, self.struct_rights)
        self.label_feature_layer = self.gather_lstm_feature(
            self.label_fwd_lstm_layer, self.label_back_lstm_layer, self.label_lefts, self.label_rights)
        
        self.struct_scores = self.evaluate_struct(
            self.struct_feature_layer, self.keep_prob)
        self.label_scores = self.evaluate_label(
            self.label_feature_layer, self.keep_prob)
        #self.struct_scores = tf.Print(struct_scores, [struct_scores], message='struct_scores:')
        #self.label_scores = tf.Print(label_scores, [label_scores], message='label_scores:')

        self.struct_softmax = tf.nn.softmax(self.struct_scores)
        self.label_softmax = tf.nn.softmax(self.label_scores)
        self.struct_softmax = tf.Print(self.struct_softmax, [self.struct_softmax], message='struct_softmax')
        self.label_softmax = tf.Print(self.label_softmax, [self.label_softmax], message='label_softmax')
        print(self.struct_softmax.get_shape())
        self.struct_loss = -tf.log(another_gather(self.struct_softmax, self.struct_actions))
        self.label_loss = -tf.log(another_gather(self.label_softmax, self.label_actions))
        print(self.struct_loss.get_shape())

        self.cost = tf.reduce_sum(self.struct_loss) + tf.reduce_sum(self.label_loss)
        #self.optimizer = tf.train.AdadeltaOptimizer(
        #    epsilon=1e-07, rho=0.99).minimize(self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.cost)

        self.saver = tf.train.Saver()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def save(self, save_path):
        self.saver.save(self.sess, save_path)
        with open(save_path+'.config', 'w') as f:
            f.write('\n')
            f.write('word_count = {}\n'.format(self.word_count))
            f.write('tag_count = {}\n'.format(self.tag_count))
            f.write('word_dims = {}\n'.format(self.word_dims))
            f.write('tag_dims = {}\n'.format(self.tag_dims))
            f.write('lstm_units = {}\n'.format(self.lstm_units))
            f.write('hidden_units = {}\n'.format(self.hidden_units))
            f.write('struct_out = {}\n'.format(self.struct_out))
            f.write('label_out = {}\n'.format(self.label_out))
            f.write('max_seq_len = {}\n'.format(self.max_seq_len))


    @staticmethod
    def load(load_path):
        with open(load_path+'.config') as f:
            f.readline()
            word_count = int(f.readline().split()[-1])
            tag_count = int(f.readline().split()[-1])
            word_dims = int(f.readline().split()[-1])
            tag_dims = int(f.readline().split()[-1])
            lstm_units = int(f.readline().split()[-1])
            hidden_units = int(f.readline().split()[-1])
            struct_out = int(f.readline().split()[-1])
            label_out = int(f.readline().split()[-1])
            max_seq_len = int(f.readline().split()[-1])
        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=struct_out,
            label_out=label_out,
            max_seq_len=max_seq_len
        )
        network.saver.restore(network.sess, load_path)
        return network

    def evaluate_recurrent(self, word_inds, tag_inds, seq_len, keep_prob):

        #[b, t, h]
        wordvec = tf.nn.embedding_lookup(self.word_embed, word_inds)
        tagvec = tf.nn.embedding_lookup(self.tag_embed, tag_inds)
        #[b, t, 2h]
        sentence = tf.concat([wordvec, tagvec], axis=-1)
        #[b, t, l]
        fwd1_out, _ = tf.nn.dynamic_rnn(
            self.fwd1, sentence, dtype=tf.float32, sequence_length=seq_len, scope='fwd1')
        back1_out, _ = tf.nn.dynamic_rnn(
            self.back1, tf.reverse_sequence(sentence, seq_lengths=seq_len, seq_axis=1, batch_dim=0), dtype=tf.float32, sequence_length=seq_len, scope='back1')

        #[b, t, 2l]
        lstm2_input = tf.concat(
            [fwd1_out, tf.reverse_sequence(back1_out, seq_lengths=seq_len, seq_axis=1, batch_dim=0)], axis=-1)

        fwd2_drop = rnn.DropoutWrapper(self.fwd2, input_keep_prob=keep_prob)
        fwd2_out, _ = tf.nn.dynamic_rnn(
            fwd2_drop, lstm2_input, dtype=tf.float32, sequence_length=seq_len, scope='fwd2')
        back2_drop = rnn.DropoutWrapper(self.back2, input_keep_prob=keep_prob)
        back2_out, _ = tf.nn.dynamic_rnn(
            back2_drop, tf.reverse_sequence(lstm2_input, seq_lengths=seq_len, seq_axis=1, batch_dim=0), dtype=tf.float32, sequence_length=seq_len, scope='back2')

        #[b, t, 2l]
        fwd_out = tf.concat([fwd1_out, fwd2_out], axis=-1)
        back_out = tf.reverse_sequence(tf.concat([back1_out, back2_out], axis=-1), seq_lengths=seq_len, seq_axis=1, batch_dim=0)

        return fwd_out, back_out

    def gather_lstm_feature(self, fwd_out, back_out, lefts, rights):
        fwd_span_vec = gather_axis(fwd_out, rights) - gather_axis(fwd_out, lefts-tf.ones_like(lefts))
        back_span_vec = gather_axis(back_out, lefts) - gather_axis(back_out, rights+tf.ones_like(rights))
        #[b, k, 4l]
        hidden_output = tf.concat([fwd_span_vec, back_span_vec], axis=-1)
        return tf.reshape(hidden_output, [-1, tf.shape(hidden_output)[1]*tf.shape(hidden_output)[2]])

    def evaluate_struct(self, hidden, keep_prob):
        uniform_init = tf.random_uniform_initializer(-0.01, 0.01)
        cons_init = tf.constant_initializer(0)
        self.W1_struct = tf.get_variable('w1_struct', shape=[4*self.struct_spans*self.lstm_units, self.hidden_units], initializer=uniform_init)
        self.b1_struct = tf.get_variable('b1_struct', shape=[self.hidden_units], initializer=cons_init)
        self.W2_struct = tf.get_variable('w2_struct', shape=[self.hidden_units, self.struct_out], initializer=cons_init)
        self.b2_struct = tf.get_variable('b2_struct', shape=[self.struct_out], initializer=cons_init)
        drop_hidden = tf.nn.dropout(hidden, keep_prob)
        hidden_output = tf.nn.relu(tf.matmul(
                drop_hidden, self.W1_struct) + self.b1_struct)
        scores = tf.matmul(hidden_output, self.W2_struct) + self.b2_struct
        return scores

    def evaluate_label(self, hidden, keep_prob):
        uniform_init = tf.random_uniform_initializer(-0.01, 0.01)
        cons_init = tf.constant_initializer(0)
        self.W1_label = tf.get_variable('w1_label', shape=[4*self.label_spans*self.lstm_units, self.hidden_units], initializer=uniform_init)
        self.b1_label = tf.get_variable('b1_label', shape=[self.hidden_units], initializer=cons_init)
        self.W2_label = tf.get_variable('w2_label', shape=[self.hidden_units, self.label_out], initializer=cons_init)
        self.b2_label = tf.get_variable('b2_label', shape=[self.label_out], initializer=cons_init)
        drop_hidden = tf.nn.dropout(hidden, keep_prob)
        hidden_output = tf.nn.relu(tf.matmul(
            drop_hidden, self.W1_label) + self.b1_label)
        scores = tf.matmul(hidden_output, self.W2_label) + self.b2_label
        return scores

    @staticmethod
    def train(
        feature_mapper,
        word_dims,
        tag_dims,
        lstm_units,
        hidden_units,
        epochs,
        batch_size,
        train_data_file,
        dev_data_file,
        model_save_file,
        droprate,
        unk_param,
        alpha=1.0,
        beta=0.0,
    ):
        print('start taining! ^_^')
        point_1 = time.clock()
        fm = feature_mapper
        word_count = fm.total_words()
        tag_count = fm.total_tags()
        training_data, max_seq_len = fm.gold_data_from_file(train_data_file)
        num_batches = len(training_data) // batch_size
        dev_trees = PhraseTree.load_treefile(dev_data_file)
        point_2 = time.clock()
        print('finish prepare train data! %f' % (point_2-point_1))
        network = Network(
            word_count=word_count,
            tag_count=tag_count,
            word_dims=word_dims,
            tag_dims=tag_dims,
            lstm_units=lstm_units,
            hidden_units=hidden_units,
            struct_out=2,
            label_out=fm.total_label_actions(),
            max_seq_len=max_seq_len
        )
        point_3 = time.clock()
        print('finish construct network! %f' % (point_3-point_2))
        parse_step = -(-num_batches // 4)
        loss_show_step = 10
        best_acc = FScore()
        for epoch in xrange(1000):
            np.random.shuffle(training_data)
            for b in xrange(num_batches):
                print('train %d epch %d bacth' % (epoch, b))
                bacth = training_data[(b * batch_size): ((b + 1) * batch_size)]
                gold_training_data = prapare_data(bacth, network.max_seq_len)
                train_feed_dict={
                    network.struct_word_inds:gold_training_data['struct_word'],
                    network.struct_tag_inds:gold_training_data['struct_tag'],
                    network.label_word_inds:gold_training_data['label_word'],
                    network.label_tag_inds:gold_training_data['label_tag'],
                    network.struct_seq_len:gold_training_data['struct_seq_len'],
                    network.label_seq_len:gold_training_data['label_seq_len'],
                    network.struct_lefts:gold_training_data['struct_lefts'],
                    network.struct_rights:gold_training_data['struct_rights'],
                    network.label_lefts:gold_training_data['label_lefts'],
                    network.label_rights:gold_training_data['label_rights'],
                    network.struct_actions:gold_training_data['struct_actions'],
                    network.label_actions:gold_training_data['label_actions'],
                    network.keep_prob:droprate,
                }
                network.sess.run(network.optimizer, feed_dict=train_feed_dict)
                if b % loss_show_step == 0:
                    loss = network.sess.run(network.cost, feed_dict=train_feed_dict)
                    num = len(gold_training_data['struct_actions']) + len(gold_training_data['label_actions'])
                    print('mean loss %f' % (loss/num))
                if  b % loss_show_step*2 == 0 or b == (num_batches - 1):
                    dev_acc = Parser.evaluate_corpus(
                        dev_trees,
                        fm,
                        network,
                    )
                    print(dev_acc)
                    if dev_acc > best_acc:
                        best_acc = dev_acc 
                        network.save(model_save_file)
