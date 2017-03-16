from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import re

from evaluate import exact_match_score, f1_score, metric_max_over_ground_truths

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim
        self.initial_encoder = tf.nn.rnn_cell.GRUCell(self.size)
        self.match_encoder_f = tf.nn.rnn_cell.GRUCell(self.size)
        self.match_encoder_b = tf.nn.rnn_cell.GRUCell(self.size)

    class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
        """Wrapper around our RNN cell implementation that allows us to play
        nicely with TensorFlow.
        """
        def __init__(self, state_size, attention_func, cell):
            # self.input_size = input_size
            self._state_size = state_size
            self.attention_func = attention_func
            self.GRUcell = cell

        @property
        def state_size(self):
            return self._state_size

        @property
        def output_size(self):
            return self._state_size

        def __call__(self, inputs, state, scope=None):
            """Updates the state using the previous @state and @inputs.
            Args:
                inputs: is the input vector of size [None, self.input_size]
                state: is the previous state vector of size [None, self.state_size]
                scope: is the name of the scope to be used when defining the variables inside.
            Returns:
                a pair of the output vector and the new state vector.
            """
            scope = scope or type(self).__name__
            attWeight = self.attention_func(inputs, state)
            inp = tf.concat(2, [state, attWeight])
            _, new_state = self.GRUCell(inp, state)
            output = new_state
            return output, new_state

    def give_attn_func(self, Hq):
        def attn_func(inputs, state):
            with tf.variable_scope("attn_func_encode", \
                initializer=tf.contrib.layers.xavier_initializer(), reuse=True):
                ### YOUR CODE HERE (~6-10 lines)
                Wq = tf.get_variable("W_q", (self.size, self.size))
                Wp = tf.get_variable("W_p", (self.size, self.size))
                Wm = tf.get_variable("W_m", (self.size, self.size))
                bp = tf.get_variable("b_p", (self.size,),\
                 initializer=tf.constant_initializer(0))
                w = tf.get_variable("w_att", (self.size, ))
                mid_st = tf.nn.tanh(tf.matmul(Hq,Wq) + tf.matmul(state,Wm)\
                 + tf.matmul(inputs, Wp) + bp)
                b = tf.get_variable("b", (1,),\
                    initializer=tf.constant_initializer(0))
                return tf.matmul(tf.softmax(tf.matmul(mid_st, w) + b), Hq)
        with tf.variable_scope("attn_func_encode", \
            initializer=tf.contrib.layers.xavier_initializer()):
            ### YOUR CODE HERE (~6-10 lines)
            Wq = tf.get_variable("W_q", (self.size, self.size))
            Wp = tf.get_variable("W_p", (self.size, self.size))
            Wm = tf.get_variable("W_m", (self.size, self.size))
            bp = tf.get_variable("b_p", (self.size,),\
             initializer=tf.constant_initializer(0))
            w = tf.get_variable("w_att", (self.size, ))
            b = tf.get_variable("b", (1,),\
                    initializer=tf.constant_initializer(0))
        return attn_func

    def encode(self, paragraph, question, masks=None, encoder_state_input=None):
        _, para_stat = tf.nn.dynamic_rnn(self.initial_encoder, paragraph, dtype=tf.float32)
        _, q_stat = tf.nn.dynamic_rnn(self.initial_encoder, question, dtype=tf.float32)
        attn_func = self.give_attn_func(q_stat)
        forward = \
        self.MatchGRUCell(self.size, attn_func, self.match_encoder_f)
        backward = \
        self.MatchGRUCell(self.size, attn_func, self.match_encoder_b)
        _, state = tf.nn.bidirectional_dynamic_rnn(forward, backward, inputs)
        state = tf.concat(state, 2)
        return state


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        forward = tf.contrib.rnn.LSTMCell(self.output_size)
        backward = tf.contrib.rnn.LSTMCell(self.output_size)
        output, _ = tf.nn.bidirectional_dynamic_rnn(forward, backward, knowledge_rep)
        return tf.split(output, 2, axis=2)

class QASystem(object):
    def __init__(self, encoder, decoder, embed_path, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        #==========Config Variables========#
        self.lr = 0.001
        self.max_para = 100
        self.max_ques = 10
        self.batch_size = 1

        # ==== set up placeholder tokens ========
        self.paragraph = tf.placeholder(tf.int32)
        self.question = tf.placeholder(tf.int32)
        # self.dropout_placeholder = tf.placeholder(tf.float32)
        self.pretrained_embeddings = tf.Variable(np.load(embed_path)['glove'], dtype=tf.float32)
        self.label_start_placeholder = tf.placeholder(tf.float32)
        self.label_end_placeholder = tf.placeholder(tf.float32)
        self.vocab_dim = encoder.vocab_dim

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system(encoder, decoder)
            self.loss = self.setup_loss()

        # ==== set up training/updating procedure ====
        # pass
        learning_rate = tf.train.exponential_decay(self.lr, global_step, 100000, 0.96)
        global_step = tf.Variable(0, trainable=False)
        t_opt=tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = t_opt.minimize(loss, global_step=global_step)
        

    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        knowledge_rep = encoder.encode(self.para_embeddings, self.ques_embeddings)
        self.start_token_score, self.end_token_score = decoder.decode(knowledge_rep)
        # raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            loss1 = tf.nn.sigmoid_cross_entropy_with_logits(self.label_start_placeholder, self.start_token_score)
            loss2 = tf.nn.sigmoid_cross_entropy_with_logits(self.label_end_placeholder, self.end_token_score)
            loss1 = tf.multiply(loss1, tf.to_float(self.mask_placeholder))
            loss2 = tf.multiply(loss2, tf.to_float(self.mask_placeholder))
            return tf.reduce_sum(loss1 + loss2)
            # temp1s = tf.multiply(tf.exp(self.start_token_soft), self.mask_placeholder)
            # temp2 = tf.multiply(tf.exp(self.end_token_soft), self.mask_placeholder)
            # loss = tf.reduce_sum(tf.multiply(tf.label_start_placeholder,temp1))/tf.reduce_sum(temp1)
            # loss += tf.reduce_sum(tf.multiply(tf.label_end_placeholder,temp2))/tf.reduce_sum(temp2)
            # pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            para_embedding_list = tf.Variable(self.pretrained_embeddings)
            para_embeddings = tf.nn.embedding_lookup(para_embedding_list, self.paragraph)
            self.para_embeddings = tf.reshape(para_embeddings, (-1, self.max_para, self.vocab_dim))
            ques_embedding_list = tf.Variable(self.pretrained_embeddings)
            ques_embeddings = tf.nn.embedding_lookup(ques_embedding_list, self.question)
            self.ques_embeddings = tf.reshape(ques_embeddings, (-1, self.max_ques, self.vocab_dim))
            # pass

    def optimize(self, session, question, paragraph, start, end):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {self.paragraph:paragraph,
        self.question:question,
        self.label_start_placeholder:start,
        self.label_end_placeholder:end}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.train_op, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, item):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        paragraph= item['context']
        question = item['question']
        input_feed = {self.paragraph:paragraph,
        self.question:question,
        self.label_start_placeholder:start,
        self.label_end_placeholder:end}
        yp, yp2 = self.decode(session, paragraph, question)
        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, paragraph, question):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {self.paragraph:paragraph,
        self.question:question}


        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.start_token_score, self.end_token_score]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, item):
        paragraph= item['context']
        question = item['question']
        yp, yp2 = self.decode(session, paragraph, question)

        a_s = np.argmax(yp[item['context_mask']], axis=1)
        a_e = np.argmax(yp2[item['context_mask']], axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for item in valid_dataset:
          valid_cost += self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, vocab, sample=100, log=True):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.
        for itr in np.random.randint(len(dataset.shape[0]), size=sample):
            item = dataset[i]
            start, end = answer(session, item)
            ans = ' '.join([vocab[x] for x in item['context'][start:end+1] if x in vocab])
            check = [' '.join([vocab[x] for x in lst]) for lst in item['span']]
            f1 += metric_max_over_ground_truths(f1_score, ans, check)
            em += metric_max_over_ground_truths(exact_match_score, ans, check)
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset, datasetVal, rev_vocab, train_dir):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        i = 0
        for itr in range(100):
            for item in dataset:
                loss_out = self.optimize(session, question, paragraph, start, end)
                i += 1
                if i % 1000:
                    print("[Sample] loss_out: %.8f " % (loss_out))
                    f1, em = self.evaluate_answer(session, datasetVal, rev_vocab)

            self.checkpoint_dir = "match_lstm"
            model_name = "match_lstm.model-epoch"
            model_dir = "squad_%s" % (self.batch_size)
            checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            self.saver.save(self.sess,
                           os.path.join(checkpoint_dir, model_name),
                           global_step=itr)