import os
import pickle

train_dir = os.path.join(os.path.curdir, 'yelp')
data_dir = os.path.join(train_dir, 'data')

for dir in [train_dir, data_dir]:
  if not os.path.exists(dir):
    os.makedirs(dir)

trainset_fn = os.path.join(data_dir, 'train.dataset')
devset_fn = os.path.join(data_dir, 'dev.dataset')
testset_fn = os.path.join(data_dir, 'test.dataset')
vocab_fn = os.path.join(data_dir, 'vocab.pickle')

reserved_tokens = 5
unknown_id = 2

vocab_size = 50000

def _read_dataset(fn, epochs=-1):
  c = 0
  while 1:
    c += 1
    if epochs > 0 and c > epochs:
      return
    print('epoch %s' % c)
    with open(fn, 'rb') as f:
      try:
        while 1:
          x, y = pickle.load(f)
          y -= 1
          assert y >= 0 and y <= 4
          yield x, y
      except EOFError:
        continue

def read_trainset(epochs=-1):
  return _read_dataset(trainset_fn, epochs=epochs)

def read_devset(epochs=-1):
  return _read_dataset(devset_fn, epochs=epochs)

def read_vocab():
  with open(vocab_fn, 'rb') as f:
    return pickle.load(f)

def read_labels():
  return {i: i for i in range(5)}
