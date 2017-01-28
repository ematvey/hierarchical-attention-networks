import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['train', 'model_lab'])
parser.add_argument("--train-dir", default=os.path.expanduser('~/yelp-cache'), dest='train_dir')
parser.add_argument("--yelp-review-file", dest='review_path')
parser.add_argument("--checkpoint-freq", type=int, dest='checkpoint_frequency', default=100)
parser.add_argument("--batch-size", type=int, dest='batch_size', default=10)
parser.add_argument("--device", default="/cpu:0")
args = parser.parse_args()

review_path = args.review_path
train_dir = args.train_dir
checkpoint_frequency = args.checkpoint_frequency

train_fn = os.path.join(args.train_dir, 'train.dataset')
dev_fn = os.path.join(args.train_dir, 'dev.dataset')
test_fn = os.path.join(args.train_dir, 'test.dataset')

tflog_dir = os.path.join(train_dir, 'tflog')

import ujson
import spacy
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from data_util import batch

import tensorflow as tf
try:
  from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
except ImportError:
  MultiRNNCell = tf.nn.rnn_cell.MultiRNNCell
  GRUCell = tf.nn.rnn_cell.GRUCell
import model


class DataIterator():
  def __init__(self):
    self.train_f = open(train_fn, 'rb')
    self.dev_f = open(dev_fn, 'rb')
    self.test_f = open(test_fn, 'rb')
  def _rotate_train(self):
    self.train_f.close()
    self.train_f = open(train_fn, 'rb')
  def _rotate_dev(self):
    self.dev_f.close()
    self.dev_f = open(dev_fn, 'rb')
  def _rotate_test(self):
    self.test_f.close()
    self.test_f = open(test_fn, 'rb')
  def _next_n(self, n, f, rotate_fn):
    x = []
    y = []
    for _ in range(n):
      try:
        x_, y_ = pickle.load(f())
      except EOFError:
        rotate_fn()
        x_, y_ = pickle.load(f())
      x.append(x_)
      y.append(y_)
    return x, y
  def train_batch(self, n):
    return self._next_n(n, lambda: self.train_f, self._rotate_train)
  def dev_batch(self, n):
    return self._next_n(n, lambda: self.dev_f, self._rotate_dev)
  def test_batch(self, n):
    return self._next_n(n, lambda: self.test_f, self._rotate_test)
  def close(self):
    self.train_f.close()
    self.dev_f.close()
    self.test_f.close()

checkpoint_name = 'yelp-model'
checkpoint_dir = os.path.join(train_dir, 'checkpoints')
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

def create_model():
  cell = GRUCell(50)
  cell = MultiRNNCell([cell]*3)
  return model.TextClassifierModel(
    vocab_size=50000, embedding_size=300, classes=5,
    word_cell=cell, sentence_cell=cell,
    word_output_size=100, sentence_output_size=100,
    device=args.device,
    max_grad_norm=15.0, dropout_keep_proba=0.5,
  )

def train():
  di = DataIterator()
  tf.reset_default_graph()
  m = create_model()

  config = tf.ConfigProto()
  # config.log_device_placement = True
  config.allow_soft_placement = True
  # config.gpu_options.allow_growth = True

  with tf.Session(config=config) as s:
    # gv = tf.global_variables()
    # v = [v for v in gv if v.op.name == 'tcm/global_step'][0]
    # gv = [v for v in gv if v.op.name != 'tcm/global_step']
    # saver = tf.train.Saver(gv)
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(tflog_dir)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint:
      print("Reading model parameters from %s" % checkpoint_path)
      saver.restore(s, checkpoint_path)
    else:
      print("Created model with fresh parameters")
      s.run(tf.global_variables_initializer())
    # s.run(tf.variables_initializer([v]))
    # saver = tf.train.Saver(tf.global_variables())
    tf.get_default_graph().finalize()
    for i in range(100000):
      x, y = di.train_batch(args.batch_size)
      y = [e-1 for e in y]
      fd = m.get_feed_data(x, y)
      step, summaries, loss, accuracy, _ = s.run([
        m.global_step,
        m.summary_op,
        m.loss,
        m.accuracy,
        m.train_op], fd)
      summary_writer.add_summary(summaries, global_step=step)
      if i % 1 == 0:
        print('step %s, loss=%s, accuracy=%s' % (step, loss, accuracy))
      if i != 0 and i % checkpoint_frequency == 0:
        print('checkpoint')
        saver.save(s, checkpoint_path)
        print('checkpoint done')
    # except Exception as e:
    #   print('saving...')
    #   saver.save(s, checkpoint_path)
    #   os.exit(1)

def main():
  if args.mode == 'train':
    train()

  elif args.mode == 'model_lab':
    tf.reset_default_graph()
    s = tf.InteractiveSession()
    m = create_model()
    import IPython
    IPython.embed()

if __name__ == '__main__':
  main()
