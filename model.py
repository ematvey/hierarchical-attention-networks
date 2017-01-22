import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, GRUCell


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states"""
    with tf.variable_scope(scope or "BidirectionalRNN") as scope:
        ((fw_outputs,
          bw_outputs),
         (fw_state,
          bw_state)) = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32))
        outputs = tf.concat_v2((fw_outputs, fw_outputs), 2)
        if isinstance(fw_state, LSTMStateTuple):
            state_c = tf.concat_v2(
                (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
            state_h = tf.concat_v2(
                (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
            state = LSTMStateTuple(c=state_c, h=state_h)
        elif isinstance(fw_state, tf.Tensor):
            state = tf.concat_v2((fw_state, bw_state), 1,
                                 name='bidirectional_concat')
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

    with tf.variable_scope(scope or 'Attention') as scope:
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
                 debug=False):
        self.debug = debug
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.classes = classes
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.sentence_cell = sentence_cell
        self.sentence_output_size = sentence_output_size

        if self.debug:
            self._init_inputs_debug()
        else:
            self._init_inputs()

        (self.document_size,
         self.sentence_size,
         self.word_size) = tf.unstack(tf.shape(self.inputs))

        self._init_embedding()
        self._init_body()
        self._init_training()

    def _init_inputs(self):
        # [document x sentence x word]
        self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32)

        # [document x sentence]
        self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32)

        # [document]
        self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32)

        # [document]
        self.labels = tf.placeholder(shape=(None,), dtype=tf.int32)

    def _init_inputs_debug(self):
        self.inputs = tf.constant([[
            [5, 4, 1, 0],
            [3, 3, 6, 7],
            [6, 7, 0, 0]
        ],
            [
            [2, 2, 1, 0],
            [3, 3, 6, 7],
            [0, 0, 0, 0]
        ]], dtype=tf.int32)

        self.word_lengths = tf.constant([
            [3, 4, 2],
            [3, 4, 0],
        ], dtype=tf.int32)

        self.sentence_lengths = tf.constant([3, 2], dtype=tf.int32)

        self.labels = tf.constant([0, 1], dtype=tf.int32)

    def _init_embedding(self):
        with tf.variable_scope("Embedding") as scope:
            self.embedding_matrix = tf.get_variable(
                name="embedding_matrix",
                shape=[self.vocab_size, self.embedding_size],
                initializer=layers.xavier_initializer(),
                dtype=tf.float32)
            self.inputs_embedded = layers.embedding_lookup_unique(
                self.embedding_matrix, self.inputs)

    def _init_body(self):
        word_level_inputs = tf.reshape(self.inputs_embedded, [
            self.document_size * self.sentence_size,
            self.word_size,
            self.embedding_size
        ])
        word_level_lengths = tf.reshape(
            self.word_lengths, [self.document_size * self.sentence_size])

        with tf.variable_scope('Word') as scope:
            word_encoder_output, _ = bidirectional_rnn(
                self.word_cell, self.word_cell,
                word_level_inputs, word_level_lengths,
                scope=scope)

            with tf.variable_scope('Attention') as scope:
                word_level_output = task_specific_attention(
                    word_encoder_output,
                    self.word_output_size,
                    scope=scope)

        # sentence_level

        sentence_inputs = tf.reshape(
            word_level_output, [self.document_size, self.sentence_size, self.word_output_size])

        with tf.variable_scope('Sentence') as scope:
            sentence_encoder_output, _ = bidirectional_rnn(
                self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

            with tf.variable_scope('Attention') as scope:
                sentence_level_output = task_specific_attention(
                    sentence_encoder_output, self.sentence_output_size, scope=scope)

        with tf.variable_scope('Classifier'):
            self.logits = layers.fully_connected(
                sentence_level_output, self.classes, activation_fn=None)

    def _init_training(self):
        with tf.variable_scope('Training'):
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.labels, logits=self.logits))
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    model = TextClassifierModel(debug=True)
    session.run(tf.global_variables_initializer())

    print(model.logits.eval())
    session.run(model.train_op)