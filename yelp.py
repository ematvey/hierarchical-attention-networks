import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['make_data', 'train', 'model_lab'])
parser.add_argument("--train-dir", default=os.path.expanduser('~/yelp-cache'), dest='train_dir')
parser.add_argument("--yelp-review-file", dest='review_path')
parser.add_argument("--checkpoint-freq", type=int, dest='checkpoint_frequency', default=100)
parser.add_argument("--batch-size", type=int, dest='batch_size', default=10)
parser.add_argument("--device", default="/cpu:0")
args = parser.parse_args()

review_path = args.review_path
train_dir = args.train_dir
checkpoint_frequency = args.checkpoint_frequency

train_fn = os.path.join(train_dir, 'train_set.pickle')
dev_fn = os.path.join(train_dir, 'dev_set.pickle')
test_fn = os.path.join(train_dir, 'test_set.pickle')

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

if args.mode == 'make_data':
  en = spacy.load('en')
  en.pipeline = [en.tagger, en.parser]

def read_reviews():
  with open(review_path, 'rb') as f:
    for line in f:
      yield ujson.loads(line)

def build_word_frequency_distribution(fn='word_freq.pickle'):
  path = os.path.join(train_dir, fn)
  try:
    with open(path, 'rb') as vocab_file:
      vocab = pickle.load(vocab_file)
      print('frequency distribution loaded')
      return vocab
  except IOError:
    print('building frequency distribution')
  def dump_vocab_counts(vocab):
    with open(path, 'wb') as vocab_file:
      pickle.dump(vocab, vocab_file)
  vocab = defaultdict(int)
  for i, review in enumerate(read_reviews()):
    doc = en.tokenizer(review['text'])
    for token in doc:
      vocab[token.orth_] += 1
    if i % 10000 == 0:
      dump_vocab_counts(vocab)
      print('dump at {}'.format(i))
  return vocab

def build_vocabulary(lower=3, n=50000, fn='vocab.pickle'):
  path = os.path.join(train_dir, fn)
  try:
    with open(path, 'rb') as vocab_file:
      vocab = pickle.load(vocab_file)
      print('vocabulary loaded')
      return vocab
  except IOError:
    print('building vocabulary')
  freq = build_word_frequency_distribution()
  top_words = list(sorted(freq.items(), key=lambda x: -x[1]))[:n-lower+1]
  vocab = {}
  i = lower
  for w, freq in top_words:
    vocab[w] = i
    i += 1
  with open(path, 'wb') as vocab_file:
    pickle.dump(vocab, vocab_file)
  return vocab

UNKNOWN = 2
def make_data(split_points=(0.8, 0.9)):
  train_ratio, dev_ratio = split_points
  vocab = build_vocabulary()
  train_f = open(train_fn, 'wb')
  dev_f = open(dev_fn, 'wb')
  test_f = open(test_fn, 'wb')

  try:
    for review in tqdm(read_reviews()):
      x = []
      for sent in en(review['text']).sents:
        x.append([vocab.get(tok.orth_, UNKNOWN) for tok in sent])
      y = review['stars']

      r = random.random()
      if r < train_ratio:
        f = train_f
      elif r < dev_ratio:
        f = dev_f
      else:
        f = test_f
      pickle.dump((x, y), f)
  except KeyboardInterrupt:
    pass

  train_f.close()
  dev_f.close()
  test_f.close()


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
  cell = GRUCell(80)
  cell = MultiRNNCell([cell]*5)
  return model.TextClassifierModel(
    vocab_size=50000, embedding_size=300, classes=5,
    word_cell=cell, sentence_cell=cell,
    word_output_size=100, sentence_output_size=100, device=args.device,
    max_grad_norm=5.0, dropout_keep_proba=0.5,
    )

def train():
  di = DataIterator()
  tf.reset_default_graph()
  m = create_model()
  try:
    config = tf.ConfigProto()
    config.log_device_placement = True
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as s:
      saver = tf.train.Saver(tf.global_variables())
      summary_writer = tf.summary.FileWriter(tflog_dir)
      checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
      if checkpoint:
        print("Reading model parameters from %s" % checkpoint_path)
        saver.restore(s, checkpoint_path)
      else:
        print("Created model with fresh parameters")
        s.run(tf.global_variables_initializer())
      for i in range(100000):
        x, y = di.train_batch(args.batch_size)
        y = [e-1 for e in y]
        fd = m.get_feed_data(x, y)
        summaries, loss, _ = s.run([m.summary_op, m.loss, m.train_op], fd)
        summary_writer.add_summary(summaries, global_step=i)
        if i % 1 == 0:
          print(loss)
        if i != 0 and i % checkpoint_frequency == 0:
          print('checkpoint')
          saver.save(s, checkpoint_path)
  except KeyboardInterrupt:
    pass
  except Exception as e:
    print("error: {}".format(e))

  import IPython
  IPython.embed()

def main():
  if args.mode == 'make_data':
    make_data()
  elif args.mode == 'train':
    train()
  elif args.mode == 'model_lab':
    tf.reset_default_graph()
    s = tf.InteractiveSession()
    m = create_model()
    import IPython
    IPython.embed()

if __name__ == '__main__':
  main()
