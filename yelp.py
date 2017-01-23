import ujson
import os
import spacy
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from data_util import batch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=['make_data', 'train'])
parser.add_argument("--data-dir", dest='data_dir', default=os.path.join(os.path.abspath('.'), 'data'))
parser.add_argument("--yelp-review-file", dest='review_path')
args = parser.parse_args()

review_path = args.review_path
data_dir = args.data_dir

if args.mode == 'make_data':
  en = spacy.load('en')
  en.pipeline = [en.tagger, en.parser]

def read_reviews():
  with open(review_path, 'rb') as f:
    for line in f:
      yield ujson.loads(line)

def build_word_frequency_distribution(fn='word_freq.pickle'):
  path = os.path.join(data_dir, fn)
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
  path = os.path.join(data_dir, fn)
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

train_fn = 'train_set.pickle'
dev_fn = 'dev_set.pickle'
test_fn = 'test_set.pickle'

train_path = os.path.join(data_dir, train_fn)
dev_path = os.path.join(data_dir, dev_fn)
test_path = os.path.join(data_dir, test_fn)

UNKNOWN = 2
def make_data(split_points=(0.8, 0.9)):
  train_ratio, dev_ratio = split_points
  vocab = build_vocabulary()
  train_f = open(train_path, 'wb')
  dev_f = open(dev_path, 'wb')
  test_f = open(test_path, 'wb')

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
    self.train_f = open(train_path, 'rb')
    self.dev_f = open(dev_path, 'rb')
    self.test_f = open(test_path, 'rb')
  def _rotate_train(self):
    self.train_f.close()
    self.train_f = open(train_path, 'rb')
  def _rotate_dev(self):
    self.dev_f.close()
    self.dev_f = open(dev_path, 'rb')
  def _rotate_test(self):
    self.test_f.close()
    self.test_f = open(test_path, 'rb')
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

checkpoint_path = os.path.join(data_dir, 'checkpoints', 'checkpoint.chpt')

def train():
  di = DataIterator()
  import tensorflow as tf
  from tensorflow.contrib.rnn import GRUCell, MultiRNNCell
  import model
  tf.reset_default_graph()
  cell = GRUCell(64)
  cell = MultiRNNCell([cell]*4)
  with tf.Session() as s:
    model = model.TextClassifierModel(vocab_size=50000, embedding_size=200, classes=5,
                                      word_cell=cell, sentence_cell=cell,
                                      word_output_size=100, sentence_output_size=100)
    saver = tf.train.Saver(tf.global_variables())
    s.run(tf.global_variables_initializer())
    for i in range(1000):
      x, y = di.train_batch(40)
      y = [e-1 for e in y]
      fd = model.get_feed_data(x, y)
      loss, _ = s.run([model.loss, model.train_op], fd)
      if i % 1 == 0:
        print(loss)
      if i % 10 == 0:
        saver.save(s, checkpoint_path)

  import IPython
  IPython.embed()

def main():
  if args.mode == 'make_data':
    make_data()
  elif args.mode == 'train':
    train()

if __name__ == '__main__':
  main()