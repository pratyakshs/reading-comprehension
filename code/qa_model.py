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

from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.util import nest

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


def batch_linear(args, output_size, bias, bias_start=0.0, scope=None, name=None):
  """Linear map: concat(W[i] * args[i]), where W[i] is a variable.
  Args:
    args: a 3D Tensor with shape [batch x m x n].
    output_size: int, second dimension of W[i] with shape [output_size x m].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.
    name: (optional) variable name.
  Returns:
    A 3D Tensor with shape [batch x output_size x n] equal to
    concat(W[i] * args[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if args.get_shape().ndims != 3:
    raise ValueError("`args` must be a 3D Tensor")

  shape = args.get_shape()
  m = shape[1].value
  n = shape[2].value
  dtype = args.dtype

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    w_name = "weights_"
    if name is not None: w_name += name
    weights = vs.get_variable(
        w_name, [output_size, m], dtype=dtype)
    res = tf.map_fn(lambda x: math_ops.matmul(weights, x), args)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      b_name = "biases_"
      if name is not None: b_name += name
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          b_name, [output_size, n],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return tf.map_fn(lambda x: math_ops.add(x, biases), res)


class Encoder(object):
    def __init__(self, size, vocab_dim):
        # size of hidden state
        self.size = size
        self.vocab_dim = vocab_dim

    def encode(self, paragraph, question, masks=None, encoder_state_input=None):
        # assume paragraph and question are already embeddings
        lstm_enc = tf.nn.rnn_cell.LSTMCell(self.size)

        with tf.variable_scope('paragraph_encoder'):
            para, _ = tf.nn.dynamic_rnn(lstm_enc, paragraph, dtype=tf.float32)
            # append sentinel
            fn = lambda x: tf.concat(
                0, [x, tf.zeros([1, self.size], dtype=tf.float32)])
            para_encoding = tf.map_fn(lambda x: fn(x), para, dtype=tf.float32)

        with tf.variable_scope('question_encoder'):
            ques, _ = tf.nn.dynamic_rnn(lstm_enc, question, dtype=tf.float32)
            # append sentinel
            fn = lambda x: tf.concat(
                0, [x, tf.zeros([1, self.size], dtype=tf.float32)])
            ques_encoding = tf.map_fn(lambda x: fn(x), ques, dtype=tf.float32)
            ques_encoding = tf.tanh(batch_linear(ques_encoding, FLAGS.question_size+1, True))
            ques_variation = tf.transpose(ques_encoding, perm=[0, 2, 1])

        with tf.variable_scope('coattention'):
            # compute affinity matrix, (batch_size, context+1, question+1)
            L = tf.batch_matmul(para_encoding, ques_variation)
            # shape = (batch_size, question+1, context+1)
            L_t = tf.transpose(L, perm=[0, 2, 1])
            # normalize with respect to question
            a_q = tf.map_fn(lambda x: tf.nn.softmax(x), L_t, dtype=tf.float32)
            # normalize with respect to context
            a_c = tf.map_fn(lambda x: tf.nn.softmax(x), L, dtype=tf.float32)
            # summaries with respect to question, (batch_size, question+1, hidden_size)
            c_q = tf.batch_matmul(a_q, para_encoding)
            c_q_emb = tf.concat(1, [ques_variation, tf.transpose(c_q, perm=[0, 2 ,1])])
            # summaries of previous attention with respect to context
            c_d = tf.batch_matmul(c_q_emb, a_c, adj_y=True)
            # final coattention context, (batch_size, context+1, 3*hidden_size)
            co_att = tf.concat(2, [para_encoding, tf.transpose(c_d, perm=[0, 2, 1])])


        with tf.variable_scope('encoder'):
            # LSTM for coattention encoding
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.size)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.size)
            # compute coattention encoding
            u, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, co_att,
                sequence_length=tf.to_int64([FLAGS.para_size]*FLAGS.batch_size),
                dtype=tf.float32)
            self._u = tf.concat(2, u)

        return self._u


class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size


    def _select(self, u, pos, idx):
        u_idx = tf.gather(u, idx)
        pos_idx = tf.gather(pos, idx)
        return tf.reshape(tf.gather(u_idx, pos_idx), [-1])





    def highway_maxout(self, hidden_size, pool_size):
        """highway maxout network."""

        def maxout(inputs,
                   num_units,
                   axis=None,
                   outputs_collections=None,
                   scope=None):
            """Adds a maxout op which is a max pooling performed in filter/channel
            dimension. This can also be used after fully-connected layers to reduce
            number of features.
            Args:
                inputs: A Tensor on which maxout will be performed
                num_units: Specifies how many features will remain after max pooling at the
                    channel dimension. This must be multiple of number of channels.
                axis: The dimension where max pooling will be performed. Default is the
                    last dimension.
                outputs_collections: The collections to which the outputs are added.
                scope: Optional scope for name_scope.
            Returns:
                A `Tensor` representing the results of the pooling operation.
            Raises:
                ValueError: if num_units is not multiple of number of features.
            """
            with ops.name_scope(scope, 'MaxOut', [inputs]) as sc:
                inputs = ops.convert_to_tensor(inputs)
                shape = inputs.get_shape().as_list()
                if axis is None:
                    # Assume that channel is the last dimension
                    axis = -1
                num_channels = shape[axis]
                if num_channels % num_units:
                    raise ValueError('number of features({}) is not '
                                     'a multiple of num_units({})'
                        .format(num_channels, num_units))
                shape[axis] = -1
                shape += [num_channels // num_units]
                outputs = math_ops.reduce_max(gen_array_ops.reshape(inputs, shape), -1,
                                              keep_dims=False)
            return utils.collect_named_outputs(outputs_collections, sc, outputs)

        def _to_3d(tensor):
            if tensor.get_shape().ndims != 2:
                raise ValueError("`tensor` must be a 2D Tensor")
            m, n = tensor.get_shape()
            return tf.reshape(tensor, [m.value, n.value, 1])

        def compute(u_t, h, u_s, u_e):
            """Computes value of u_t given current u_s and u_e."""
            # reshape
            u_t = _to_3d(u_t)
            h = _to_3d(h)
            u_s = _to_3d(u_s)
            u_e = _to_3d(u_e)
            # non-linear projection of decoder state and coattention
            state_s = tf.concat(1, [h, u_s, u_e])
            r = tf.tanh(batch_linear(state_s, hidden_size, False, name='r'))
            u_r = tf.concat(1, [u_t, r])
            # first maxout
            m_t1 = batch_linear(u_r, pool_size*hidden_size, True, name='m_1')
            m_t1 = maxout(m_t1, hidden_size, axis=1)
            # second maxout
            m_t2 = batch_linear(m_t1, pool_size*hidden_size, True, name='m_2')
            m_t2 = maxout(m_t2, hidden_size, axis=1)
            # highway connection
            mm = tf.concat(1, [m_t1, m_t2])
            # final maxout
            res = maxout(batch_linear(mm, pool_size, True, name='mm'), 1, axis=1)
            return res

        return compute


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
        batch_size = FLAGS.batch_size
        hidden_size = FLAGS.state_size
        maxout_size = FLAGS.maxout_size
        max_timesteps = FLAGS.para_size
        max_decode_steps = FLAGS.max_decode_steps

        with tf.variable_scope('selector'):
            # LSTM for decoding
            lstm_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
            # init highway fn
            highway_alpha = self.highway_maxout(hidden_size, maxout_size)
            highway_beta = self.highway_maxout(hidden_size, maxout_size)
            # reshape knowledge_rep, (context, batch_size, 2*hidden_size)
            U = tf.transpose(knowledge_rep[:,:max_timesteps,:], perm=[1, 0, 2])
            # batch indices
            loop_until = tf.to_int32(np.array(range(batch_size)))
            # initial estimated positions
            s, e = tf.split(0, 2, [0, 1])

            fn = lambda idx: self._select(knowledge_rep, s, idx)
            u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

            fn = lambda idx: self._select(knowledge_rep, e, idx)
            u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)


        self._s, self._e = [], []
        self._alpha, self._beta = [], []
        with tf.variable_scope('decoder') as vs:
            for step in range(max_decode_steps):
                if step > 0: vs.reuse_variables()
                # single step lstm
                _input = tf.concat(1, [u_s, u_e])
                _, h = tf.nn.rnn(lstm_dec, [_input], dtype=tf.float32)
                h_state = tf.concat(1, h)
                with tf.variable_scope('highway_alpha'):
                    # compute start position first
                    fn = lambda u_t: highway_alpha(u_t, h_state, u_s, u_e)
                    alpha = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                    s = tf.reshape(tf.argmax(alpha, 0), [batch_size])
                    # update start guess
                    fn = lambda idx: self._select(knowledge_rep, s, idx)
                    u_s = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)
                with tf.variable_scope('highway_beta'):
                    # compute end position next
                    fn = lambda u_t: highway_beta(u_t, h_state, u_s, u_e)
                    beta = tf.map_fn(lambda u_t: fn(u_t), U, dtype=tf.float32)
                    e = tf.reshape(tf.argmax(beta, 0), [batch_size])
                    # update end guess
                    fn = lambda idx: self._select(knowledge_rep, e, idx)
                    u_e = tf.map_fn(lambda idx: fn(idx), loop_until, dtype=tf.float32)

                self._s.append(s)
                self._e.append(e)
                self._alpha.append(tf.reshape(alpha, [batch_size, -1]))
                self._beta.append(tf.reshape(beta, [batch_size, -1]))

        return self._alpha[-1], self._beta[-1]

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
        self.max_para = FLAGS.para_size
        self.max_ques = FLAGS.question_size

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
        t_opt=tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op=t_opt.minimize(self.loss)


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
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.label_start_placeholder, self.start_token_score))
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.label_end_placeholder, self.end_token_score))
            return loss
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

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

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
        paragraph = item['context']
        question = item['question']
        mask = np.array(item['contextMask'])
        yp, yp2 = self.decode(session, paragraph, question)
        a_s = np.argmax(np.ma.masked_array(yp, ~mask), axis=1)
        a_e = np.argmax(np.ma.masked_array(yp2, ~mask), axis=1)

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

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


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
        for itr in np.random.randint(len(dataset['context']), size=sample):
            context = dataset['context'][itr]
            span = dataset['span'][itr]
            item = {key: dataset[key][itr] for key in dataset.keys()}
            start, end = self.answer(session, item)
            start = start[0]
            end = end[0]
            ans = ' '.join([vocab[x] for x in context[start:end+1] if x in vocab])
            check = ' '.join([vocab[x] for x in context[span[0]:span[1]+1]])
            print('span', span)
            print('startend', start, end)
            print('ans', ans)
            print('check', check)
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

        print(type(dataset))
        question = dataset['question']
        questionMask = dataset['questionMask']
        context = dataset['context']
        contextMask = dataset['contextMask']
        span = dataset['span']
        print('question.size', np.array(question).shape)
        print('context.size', np.array(context).shape)
        print('span.size', np.array(span).shape)
        i = 1
        for itr in range(100):
            for j in range(len(question)):
                print('iter,', itr, 'j=', j)
                print('question[i]', question[i])
                print('context[i]', context[i])
                print('span[i]', span[i])
                q = np.array(question[i]).reshape((FLAGS.question_size, 1))
                c = np.array(context[i]).reshape((FLAGS.para_size, 1))
                loss_out = self.optimize(session, question[i], context[i], span[i][0], span[i][1])
                i += 1
                if i % 1000 == 0:
                    print("[Sample] loss_out:", (loss_out))
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
