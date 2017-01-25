import tensorflow as tf
import tensorflow.contrib.layers as layers

try:
    from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell
except ImportError:
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    GRUCell = tf.nn.rnn_cell.GRUCell

import data_util


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "birnn") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            scope=scope))
        outputs = tf.concat_v2((fw_outputs, fw_outputs), 2)

        def concatenate_state(fw_state, bw_state):
            if isinstance(fw_state, LSTMStateTuple):
                state_c = tf.concat_v2(
                    (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
                state_h = tf.concat_v2(
                    (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
                state = LSTMStateTuple(c=state_c, h=state_h)
            elif isinstance(fw_state, tf.Tensor):
                state = tf.concat_v2((fw_state, bw_state), 1,
                                    name='bidirectional_concat')
            elif (isinstance(fw_state, tuple) and
                    isinstance(bw_state, tuple) and
                    len(fw_state) == len(bw_state)):
                # multilayer
                state = tuple(concatenate_state(fw, bw) for fw, bw in zip(fw_state, bw_state))

            else:
                raise ValueError('unknown state type: {}'.format((fw_state,bw_state)))

        state = concatenate_state(fw_state, bw_state)
        return outputs, state


def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).

    Args:
        inputs: Tensor of shape [batch_size, units, input_size]
            `input_size` must be static (known)
            `units` axis will be attended over (reduced from output)
            `batch_size` will be preserved
        output_size: Size of output's inner (feature) dimension

    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape(
    )[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)
        attention_weights = tf.nn.softmax(
            tf.multiply(input_projection, attention_context_vector)
        )

        weighted_projection = tf.multiply(input_projection, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs


class TextClassifierModel():
    """ Implementation of document classification model described in
        `Hierarchical Attention Networks for Document Classification (Yang et al., 2016)`
        (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)"""

    def __init__(self,
                 vocab_size=10,
                 embedding_size=5,
                 classes=2,
                 word_cell=GRUCell(10),
                 sentence_cell=GRUCell(10),
                 word_output_size=10,
                 sentence_output_size=10,
                 max_grad_norm=5.0,
                 scope=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size
        self.max_grad_norm = max_grad_norm

        with tf.variable_scope(scope or 'tcm') as scope:
            self._init_inputs(scope)

            (self.document_size,
            self.sentence_size,
            self.word_size) = tf.unstack(tf.shape(self.inputs))

            self._init_embedding(scope)
            self._init_body(scope)

        with tf.variable_scope('train'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits))

            tf.summary.scalar('loss', self.loss)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(1e-4)

            self.train_op = opt.apply_gradients(zip(grads, tvars), name='train_op')

            self.summary_op = tf.summary.merge_all()

    def _init_inputs(self, scope):
        with tf.variable_scope(scope) as scope:
            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32)

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32)

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32)

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32)

    def _init_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.embedding_matrix = tf.get_variable(
                    name="embedding_matrix",
                    shape=[self.vocab_size, self.embedding_size],
                    initializer=layers.xavier_initializer(),
                    dtype=tf.float32)
                self.inputs_embedded = layers.embedding_lookup_unique(
                    self.embedding_matrix, self.inputs)

    def _init_body(self, scope):
        with tf.variable_scope(scope):

            word_level_inputs = tf.reshape(self.inputs_embedded, [
                self.document_size * self.sentence_size,
                self.word_size,
                self.embedding_size
            ])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.document_size * self.sentence_size])

            with tf.variable_scope('word') as scope:
                word_encoder_output, _ = bidirectional_rnn(
                    self.word_cell, self.word_cell,
                    word_level_inputs, word_level_lengths,
                    scope=scope)

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

            # sentence_level

            sentence_inputs = tf.reshape(
                word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence') as scope:
                sentence_encoder_output, _ = bidirectional_rnn(
                    self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, self.sentence_output_size, scope=scope)

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sentence_level_output, self.classes, activation_fn=None)

    def get_feed_data(self, x, y=None):
        x_m, doc_sizes, sent_sizes = data_util.batch(x)
        fd = {
            self.inputs: x_m,
            self.sentence_lengths: doc_sizes,
            self.word_lengths: sent_sizes,
        }
        if y is not None:
            fd[self.labels] = y
        return fd


if __name__ == '__main__':
    tf.reset_default_graph()
    with tf.Session() as session:
        model = TextClassifierModel()
        session.run(tf.global_variables_initializer())

        fd = {
            model.inputs: [[
                [5, 4, 1, 0],
                [3, 3, 6, 7],
                [6, 7, 0, 0]
            ],
                [
                [2, 2, 1, 0],
                [3, 3, 6, 7],
                [0, 0, 0, 0]
            ]],
            model.word_lengths: [
                [3, 4, 2],
                [3, 4, 0],
            ],
            model.sentence_lengths: [3, 2],
            model.labels: [0, 1],
        }

        print(session.run(model.logits, fd))
        session.run(model.train_op, fd)