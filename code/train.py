from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_data import PAD_ID
from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 1, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 80, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 150, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 2, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 500, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("question_size", 60, "Size of q (default 60)")
tf.app.flags.DEFINE_integer("para_size", 800, "The para size (def 800)")
# tf.app.flags.DEFINE_string("checkpoint_dir", "match_gru", "Directory to save match_gru (def: match_gru)")
tf.app.flags.DEFINE_integer("trainable", 0, "training embed?")
tf.app.flags.DEFINE_integer("current_ep", 0, "current_ep")


FLAGS = tf.app.flags.FLAGS


def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir


def init_dataset(data_dir, val=False):
    if val:
        qfile = pjoin(data_dir, 'val.ids.question')
        cfile = pjoin(data_dir, 'val.ids.context')
        sfile = pjoin(data_dir, 'val.span')
    else:
        qfile = pjoin(data_dir, 'train.ids.question')
        cfile = pjoin(data_dir, 'train.ids.context')
        sfile = pjoin(data_dir, 'train.span')

    dataset_dicts = {'question': [], 'questionMask': [], 'context': [],
                     'contextMask': [], 'contextLen': [], 
                     'questionLen': [], 'span_exact':[], 'span' :[]}

    with open(qfile, 'rb') as qf, open(cfile, 'rb') as cf, open(sfile, 'rb') as sf:
        for line in qf:
            question = [int(word) for word in line.strip().split()]
            context = [int(word) for word in cf.next().strip().split()]
            span = [int(word) for word in sf.next().strip().split()]
            span_min = [min(x, FLAGS.para_size - 1) for x in span]

            # do question padding
            question_len = len(question)
            if len(question) > FLAGS.question_size:
                question = question[:FLAGS.question_size]
                q_mask = [True] * FLAGS.question_size
            else:
                question = question + [PAD_ID] * (FLAGS.question_size - len(question))
                q_mask = [True] * len(question) + [False] *  (FLAGS.question_size - len(question))

            # do context padding
            para_len = len(context)
            if len(context) > FLAGS.para_size:
                context = context[:FLAGS.para_size]
                c_mask = [True] * FLAGS.para_size
            else:
                context = context + [PAD_ID] * (FLAGS.para_size - len(context))
                c_mask = [True] * len(context) + [False] *  (FLAGS.para_size - len(context))


            dataset_dicts['question'].append(question)
            dataset_dicts['questionMask'].append(q_mask)
            dataset_dicts['context'].append(context)
            dataset_dicts['contextMask'].append(c_mask)
            #st = [0 for x in range(FLAGS.para_size)]
            #st[min(span[0], self.para_size)] = 1
            #end = [0 for x in range(FLAGS.para_size)]
            #end[min(span[1], self.para_size)] = 1
            #dataset_dicts['spanStart'].append(st)
            #dataset_dicts['spanEnd'].append(end)
            dataset_dicts['span_exact'].append(span)
            dataset_dicts['span'].append(span_min)
            dataset_dicts['contextLen'].append(para_len)
            dataset_dicts['questionLen'].append(question_len)

    return dataset_dicts


def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    datasetTrain = init_dataset(FLAGS.data_dir, val=False)
    datasetVal = init_dataset(FLAGS.data_dir, val=True)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size, size=FLAGS.state_size)

    qa = QASystem(encoder, decoder, embed_path)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    gpu_options = tf.GPUOptions(allow_growth=True)
    #config=tf.ConfigProto(gpu_options=gpu_options\
    #  , allow_soft_placement=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options\
      , allow_soft_placement=True)) as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, datasetTrain, datasetVal, rev_vocab, save_train_dir)
        #FLAGS.evaluate,
        qa.evaluate_answer(sess, datasetVal, rev_vocab, log=True)

if __name__ == "__main__":
    tf.app.run()
