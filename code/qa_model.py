from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

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

        with tf.variable_scope('p_enc'):
            self.initial_p_encoder = tf.nn.rnn_cell.GRUCell(self.size)
        with tf.variable_scope('q_enc'):
            self.initial_q_encoder = tf.nn.rnn_cell.GRUCell(self.size)
        with tf.variable_scope('mf_enc'):
            self.match_encoder_f = tf.nn.rnn_cell.GRUCell(self.size)
        with tf.variable_scope('mb_enc'):
            self.match_encoder_b = tf.nn.rnn_cell.GRUCell(self.size)
        with tf.variable_scope("attn_func_encode", \
            initializer=tf.contrib.layers.xavier_initializer()):
            ### YOUR CODE HERE (~6-10 lines)
            self.Wq = tf.get_variable("W_q", (self.size, self.size))
            self.Wp = tf.get_variable("W_p", (self.size, self.size))
            self.Wm = tf.get_variable("W_m", (self.size, self.size))
            self.bp = tf.get_variable("b_p", (self.size,),\
             initializer=tf.constant_initializer(0))
            self.w = tf.get_variable("w_att", (self.size, 1))
            self.b = tf.get_variable("b", (1,),\
                    initializer=tf.constant_initializer(0))

    class MatchGRUCell(tf.nn.rnn_cell.RNNCell):
        """Wrapper around our RNN cell implementation that allows us to play
        nicely with TensorFlow.
        """
        def __init__(self, state_size, attention_func, cell):
            # self.input_size = input_size
            self._state_size = state_size
            self.attention_func = attention_func
            self.GRUCell = cell

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
            # print("inputs ")
            # print(inputs.get_shape())
            inp = tf.concat(1, [inputs, attWeight])
            # print("inpot")
            # print(inp.get_shape())
            output, new_state = self.GRUCell(inp, state)
            # output = new_state
            return output, new_state

    def give_attn_func(self, Hq, mask):
        def attn_func(inputs, state):
            # print(Hq.get_shape())
            res = tf.reshape(Hq,[-1,int(Hq.get_shape()[2])])
            temp = tf.matmul(res,self.Wq)
            other_fac = tf.tile(tf.reshape(tf.matmul(state,self.Wm)+ tf.matmul(inputs, self.Wp) \
            + self.bp, [-1, 1, self.size]),[1,int(Hq.get_shape()[1]),1])
            # print(other_fac.get_shape())
            # print(temp.get_shape())
            mid_st = tf.matmul(tf.reshape(tf.nn.tanh(\
               tf.reshape(temp, [-1, int(Hq.get_shape()[1]),self.size])\
                 + other_fac),\
                  [-1,int(Hq.get_shape()[2])]), self.w) + self.b
            # print("mid checkk")
            # print(mid_st.get_shape())
            mid_st=tf.multiply(tf.to_float(mask), tf.exp(tf.reshape(mid_st, [-1, int(Hq.get_shape()[1])])))
            mid_st=mid_st/tf.reduce_sum(mid_st, axis=1, keep_dims=True)
            # tf.nn.softmax(tf.matmul(mid_st, self.w) + self.b)
            mid_st = tf.reshape(tf.batch_matmul(tf.reshape(mid_st,\
            [-1, 1,int(Hq.get_shape()[1])]), Hq),\
            [-1, int(Hq.get_shape()[2])])
            # print("asser")
            # print(mid_st.get_shape())
            return mid_st
        return lambda x, y: attn_func(x, y) 

    def encode(self, paragraph, question, masks, question_len, paragraph_len):
        with tf.variable_scope('p_enc'):
            para_stat, _ = tf.nn.dynamic_rnn(self.initial_p_encoder, paragraph,\
                sequence_length=paragraph_len, dtype=tf.float32)
        with tf.variable_scope('q_enc'):
            q_stat, _ = tf.nn.dynamic_rnn(self.initial_q_encoder, question,\
            sequence_length=question_len, dtype=tf.float32)
        attn_func = self.give_attn_func(q_stat, masks)
        # print("GEEE")
        with tf.variable_scope('mf_enc'):
            forward = \
            self.MatchGRUCell(self.size, attn_func, self.match_encoder_f)
        with tf.variable_scope('mb_enc'):
            backward = \
            self.MatchGRUCell(self.size, attn_func, self.match_encoder_b)
        # print("Para")        
        # print(para_stat.get_shape())
        # print("Lala")        
        # print(paragraph.get_shape())
        # paragraph_len = tf.Print(paragraph_len, [paragraph_len])
        state, _ = tf.nn.bidirectional_dynamic_rnn(forward, backward, \
        para_stat, sequence_length=paragraph_len, dtype=tf.float32)
        state = tf.concat(2, state)
        # print(state.get_shape())
        return state


class Decoder(object):
    def __init__(self, output_size, size):
        self.output_size = output_size
        self.size = size

    def decode(self, knowledge_rep, mask):
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
        with tf.variable_scope("attn_func_decode_start", \
            initializer=tf.contrib.layers.xavier_initializer()):
            ### YOUR CODE HERE (~6-10 lines)
            V = tf.get_variable("V", (2*self.size, self.size))
            ba = tf.get_variable("b_a", (self.size,),\
             initializer=tf.constant_initializer(0))
            v = tf.get_variable("v_att", (self.size, 1))
            c = tf.get_variable("c", (1,),\
                    initializer=tf.constant_initializer(0))
            Hq = knowledge_rep
            res = tf.reshape(Hq,[-1,int(Hq.get_shape()[2])])
            temp = tf.matmul(res,V)
            # self.start_logits = tf.reshape(tf.matmul(tf.reshape(tf.nn.tanh(\
            # tf.reshape(temp, [-1, int(Hq.get_shape()[1]),self.size]) + ba),\
            #   [-1,self.size]), v) + c, [-1, int(Hq.get_shape()[1])])
            self.start_logits = tf.reshape(tf.matmul(tf.nn.tanh(\
            temp + ba), v) + c, [-1, int(Hq.get_shape()[1])])
            # print(self.start_logits.get_shape())
            mid_st=tf.multiply(tf.to_float(mask), tf.exp(self.start_logits))
            mid_st=mid_st/tf.reduce_sum(mid_st, axis=1, keep_dims=True)
            attention_mat = tf.batch_matmul(tf.reshape(tf.nn.softmax(\
            mid_st),[-1, 1, int(Hq.get_shape()[1])]), Hq)
            # print(attention_mat.get_shape())
        with tf.variable_scope("attn_func_decode_end", \
            initializer=tf.contrib.layers.xavier_initializer()):
            V = tf.get_variable("V", (2*self.size, self.size))
            Wp = tf.get_variable("W_a", (2*self.size, self.size))
            ba = tf.get_variable("b_a", (self.size,),\
             initializer=tf.constant_initializer(0))
            v = tf.get_variable("v_att", (self.size, 1))
            c = tf.get_variable("c", (1,),\
                    initializer=tf.constant_initializer(0))
            Hq = knowledge_rep
            res = tf.reshape(Hq,[-1,int(Hq.get_shape()[2])])
            temp = tf.matmul(res,V)
            other_fac = tf.tile(tf.reshape(tf.matmul(tf.reshape(attention_mat,\
            [-1, 2*self.size]), Wp), [-1, 1, self.size]),\
            [1,int(Hq.get_shape()[1]),1])
            self.end_logits = tf.reshape(tf.matmul(tf.reshape(tf.nn.tanh(\
            tf.reshape(temp, [-1, int(Hq.get_shape()[1]),self.size]) + ba +\
            other_fac), [-1,self.size]), v) + c, [-1, int(Hq.get_shape()[1])])
            # print(self.end_logits.get_shape())
            
        return self.start_logits, self.end_logits

class QASystem(object):
    def __init__(self, encoder, decoder, embed_path, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        #==========Config Variables========#
        self.lr = FLAGS.learning_rate
        self.max_para = FLAGS.para_size
        self.max_ques = FLAGS.question_size

        # ==== set up placeholder tokens ========
        self.paragraph = tf.placeholder(tf.int32)
        self.question = tf.placeholder(tf.int32)
        self.paragraph_mask = tf.placeholder(tf.int32)
        self.question_mask = tf.placeholder(tf.int32)
        self.paragraph_len = tf.placeholder(tf.int32)
        self.question_len = tf.placeholder(tf.int32)
        # self.dropout_placeholder = tf.placeholder(tf.float32)
        self.pretrained_embeddings = np.load(embed_path)['glove']
        self.label_start_placeholder = tf.placeholder(tf.float32)
        self.label_end_placeholder = tf.placeholder(tf.float32)
        self.vocab_dim = encoder.vocab_dim

        # ==== assemble pieces ====
        # with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            self.setup_embeddings()
            self.setup_system(encoder, decoder)
            self.loss = self.setup_loss()

        # ==== set up training/updating procedure ====
        # pass
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.lr, global_step, 100000, 0.96)
        t_opt=tf.train.AdamOptimizer(learning_rate=learning_rate)

        grad_var_list = t_opt.compute_gradients(self.loss)
        
        grad_list, _ = tf.clip_by_global_norm([x[0] for x in grad_var_list]\
            , FLAGS.max_gradient_norm)
        grad_var_list = [(grad, pair[1]) for grad ,pair in zip(grad_list, grad_var_list)]
        self.grad_norm = tf.global_norm([x[0] for x in grad_var_list])
        self.train_op = t_opt.apply_gradients(grad_var_list, global_step=global_step)
        self.saver = tf.train.Saver(max_to_keep=1000)
        # self.train_op = t_opt.minimize(self.loss, global_step=global_step)
        

    def setup_system(self, encoder, decoder):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        knowledge_rep = encoder.encode(self.para_embeddings, self.ques_embeddings\
        , self.question_mask, self.question_len, self.paragraph_len)
        self.start_token_score, self.end_token_score = decoder.decode(knowledge_rep,\
            self.paragraph_mask)
        # raise NotImplementedError("Connect all parts of your system here!")


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            mid_st=tf.multiply(tf.to_float(self.paragraph_mask), tf.exp(self.start_token_score))
            mid_st=tf.multiply(self.label_start_placeholder,\
                tf.log(mid_st/tf.reduce_sum(mid_st, axis=1, keep_dims=True)))
            loss = tf.reduce_mean(tf.reduce_sum(mid_st, axis=1))
            mid_st=tf.multiply(tf.to_float(self.paragraph_mask), tf.exp(self.end_token_score))
            mid_st=tf.multiply(self.label_end_placeholder,\
                tf.log(mid_st/tf.reduce_sum(mid_st, axis=1, keep_dims=True)))
            loss += tf.reduce_mean(tf.reduce_sum(mid_st, axis=1))
            return -loss
            # loss1 = tf.nn.sigmoid_cross_entropy_with_logits(self.label_start_placeholder, )
            # loss2 = tf.nn.sigmoid_cross_entropy_with_logits(self.label_end_placeholder, self.end_token_score)
            # loss1 = tf.multiply(loss1, tf.to_float(self.mask_placeholder))
            # loss2 = tf.multiply(loss2, tf.to_float(self.mask_placeholder))
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
            para_embedding_list = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            para_embeddings = tf.nn.embedding_lookup(para_embedding_list, self.paragraph)
            self.para_embeddings = tf.reshape(para_embeddings, (-1, self.max_para, self.vocab_dim))
            ques_embedding_list = tf.Variable(self.pretrained_embeddings, dtype=tf.float32)
            ques_embeddings = tf.nn.embedding_lookup(ques_embedding_list, self.question)
            self.ques_embeddings = tf.reshape(ques_embeddings, (-1, self.max_ques, self.vocab_dim))
            # pass

    def optimize(self, session, question, paragraph, start, end, 
        question_mask, question_len, paragraph_mask, paragraph_len):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {self.paragraph:paragraph,
        self.question:question,
        self.label_start_placeholder:start,
        self.label_end_placeholder:end,
        self.question_len: question_len,
        self.paragraph_mask: paragraph_mask,
        self.question_mask: question_mask,
        self.paragraph_len: paragraph_len}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.train_op, self.grad_norm, self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, question, paragraph, start, end, 
        question_mask, question_len, paragraph_mask, paragraph_len):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {self.paragraph:paragraph,
        self.question:question,
        self.label_start_placeholder:start,
        self.label_end_placeholder:end,
        self.question_len: question_len,
        self.paragraph_mask: paragraph_mask,
        self.question_mask: question_mask,
        self.paragraph_len: paragraph_len}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, question, paragraph, 
        question_mask, question_len, paragraph_mask, paragraph_len):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {self.paragraph:paragraph,
        self.question:question,
        self.question_len: question_len,
        self.paragraph_mask: paragraph_mask,
        self.question_mask: question_mask,
        self.paragraph_len: paragraph_len}


        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = [self.start_token_score, self.end_token_score]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, item):
        paragraph = [item['context']]
        question = [item['question']]
        maskP = [item['contextMask']]
        maskQ = [item['questionMask']]
        yp, yp2 = self.decode(session, question, paragraph, maskQ,\
         [item['questionLen']], maskP, [item['contextLen']])
        a_s = np.argmax(np.ma.masked_array(yp, np.logical_not(maskP)), axis=1)
        a_e = np.argmax(np.ma.masked_array(yp2, np.logical_not(maskP)), axis=1)

        return (a_s, a_e)

    def validate(self, sess, dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0
        question = dataset['question']
        questionMask = dataset['questionMask']
        context = dataset['context']
        contextMask = dataset['contextMask']
        contextLen = dataset['contextLen']
        questionLen = dataset['questionLen']
        span_start = dataset['spanStart']
        span_end = dataset['spanEnd']
        batch_size = FLAGS.batch_size
        num_examples = len(question)
        num_batches = int(num_examples / batch_size) + 1
        for j in range(num_batches):
            question_batch = question[j*batch_size:(j+1)*batch_size]
            context_batch = context[j*batch_size:(j+1)*batch_size]
            start_batch = span[j*batch_size:(j+1)*batch_size]
            end_batch = span[j*batch_size:(j+1)*batch_size]
            question_mask_batch = questionMask[j*batch_size:(j+1)*batch_size]
            context_mask_batch = contextMask[j*batch_size:(j+1)*batch_size]
            question_len_batch = questionLen[j*batch_size:(j+1)*batch_size]
            context_len_batch = contextLen[j*batch_size:(j+1)*batch_size]
            
            loss_out = self.test(session, question_batch, context_batch, start_batch, end_batch,\
                question_mask_batch, question_len_batch, context_mask_batch, context_len_batch)
            valid_cost += loss_out
        # for itr in range(len(valid_dataset)):
        #     context = dataset['context'][itr]
        #     span = dataset['span'][itr]
        #     item = {key: dataset[key][itr] for key in dataset.keys()}
        #     valid_cost += self.test(sess, item)

        return valid_cost/float(num_batches)

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
        for itr in np.random.randint(len(dataset['context']), size=sample):
            context = dataset['context'][itr]
            span = dataset['span'][itr]
            item = {key: dataset[key][itr] for key in dataset.keys()}
            start, end = self.answer(session, item)
            start = start[0]
            end = end[0]
            ans = ' '.join([vocab[x] for x in context[start:end+1] if x in vocab])
            check = ' '.join([vocab[x] for x in context[span[0]:span[1]+1]])
            # print('span', span)
            # print('startend', start, end)
            # print('ans', ans)
            # print('check', check)
            f1 += metric_max_over_ground_truths(f1_score, ans, check)
            em += metric_max_over_ground_truths(exact_match_score, ans, check)
        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def load(self, session, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_name = "match_lstm.model-epoch"
        model_dir = "squad_%s" % (FLAGS.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

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
        checkpoint_dir = FLAGS.train_dir
	model_name = "match_lstm.model-epoch"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.load(session, checkpoint_dir)
        #self.saver.save(session, os.path.join(checkpoint_dir, model_name),global_step=0)
        


        question = dataset['question']
        questionMask = dataset['questionMask']
        context = dataset['context']
        contextMask = dataset['contextMask']
        contextLen = dataset['contextLen']
        questionLen = dataset['questionLen']
        span_start = dataset['spanStart']
        span_end = dataset['spanEnd']

        batch_size = FLAGS.batch_size
        num_examples = len(question)
        num_batches = int(num_examples / batch_size) + 1
        
        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        i = 0
        min_val = 10000000000000000
        min_model_name="dummy"
        for itr in range(FLAGS.epochs):
            for j in range(num_batches):
                tic =time.time()
                print('iter,', itr, 'j=', j)
                question_batch = question[j*batch_size:(j+1)*batch_size]
                context_batch = context[j*batch_size:(j+1)*batch_size]
                start_batch = span_start[j*batch_size:(j+1)*batch_size]
                end_batch = span_end[j*batch_size:(j+1)*batch_size]
                question_mask_batch = questionMask[j*batch_size:(j+1)*batch_size]
                context_mask_batch = contextMask[j*batch_size:(j+1)*batch_size]
                question_len_batch = questionLen[j*batch_size:(j+1)*batch_size]
                context_len_batch = contextLen[j*batch_size:(j+1)*batch_size]
                # print(context_len_batch)
                _, grad_norm, loss_out = self.optimize(session, question_batch, context_batch, start_batch, end_batch,\
                    question_mask_batch, question_len_batch, context_mask_batch, context_len_batch)
                print("[Sample] loss_out: %.8f , norm: %.8f" % (loss_out, grad_norm))
                i += 1
                toc=time.time()
                print("time")
                print(toc-tic)
                if i % FLAGS.print_every == 0:
                    f1, em = self.evaluate_answer(session, datasetVal, rev_vocab)
                    loss_val = self.validate(session, datasetVal)
                    print("[Sample Validate] loss_out: %.8f, F1: %.8f, EM: %.8f " % (loss_val, f1, em))

            # self.checkpoint_dir = FLAGS.train_dir
            model_name = "match_lstm.model-epoch"
            #model_dir = "squad_%s" % (FLAGS.batch_size)
            #checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            loss_val = self.validate(session, datasetVal)
            if loss_val < min_val:
                min_model_name = itr
                print("New min model itr: " + str(itr))
            self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),global_step=itr)
