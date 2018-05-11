import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell, GRUCell

from .environment import current_env


def biGRU(inputs, inputs_len, scope="Bidirectional_GRU", reuse=None, return_state=False):
    """
    Bidirectional recurrent neural network with GRU cells.

    Args:
        inputs:     rnn input of shape (batch_size, timestep, dim)
        inputs_len: rnn input_len of shape (batch_size, )
        scope:
        reuse:
        return_state:     if 0, output returns rnn output for every timestep,
                    if 1, output returns concatenated state of backward and
                    forward rnn.
    """
    env = current_env()

    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()
        if inputs.shape.ndims > 3:
            inputs = tf.reshape(inputs, (-1, shapes[-2], shapes[-1]))
            inputs_len = tf.reshape(inputs_len, (-1,))

        units = 150
        dropout = 0.2

        cell_fw = DropoutWrapper(GRUCell(units), output_keep_prob=1 - dropout)
        cell_bw = DropoutWrapper(GRUCell(units), output_keep_prob=1 - dropout)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                          sequence_length=inputs_len,
                                                          dtype=tf.float32)
        if return_state:
            return tf.reshape(tf.concat(states, 1), (env.batch_size, shapes[1], 2 * units))

        return tf.concat(outputs, 2)


class AttentionGRUCell(GRUCell):
    def __init__(self, *args, **kwargs):
        self.attention = kwargs.pop('attention_input')
        super(AttentionGRUCell, self).__init__(*args, **kwargs)

    def build(self, inputs_shape):
        input_depth = inputs_shape[1].value
        att_input_depth = self.attention.shape[-1].value

        self.weight_att = self.add_variable(
            "att/W_Q", shape=[att_input_depth, self._num_units],
            # initializer=self._kernel_initializer
        )

        self.weight_input = self.add_variable(
            'att/W_P', shape=[input_depth, self._num_units]
        )

        self.weight_gate = self.add_variable(
            'att/W_G', shape=[4 * self._num_units, 4 * self._num_units]
        )

        self.weight_score = self.add_variable(
            'att/v', shape=[self._num_units]
        )

        super(AttentionGRUCell, self).build(inputs_shape=(inputs_shape[0], tf.Dimension(2) * inputs_shape[1]))

    def att(self, inputs):
        # preparing attention-pooling vector
        # question_words = tf.reshape(
        #     tf.matmul(
        #         tf.reshape(self.attention, (-1, self.attention.shape[-1])),
        #         self.weight_att
        #     ),
        #     (self.attention.shape[0], self.attention.shape[1], -1)
        # )
        # passage_word = tf.reshape(
        #     tf.matmul(
        #         tf.reshape(inputs, (-1, inputs.shape[-1])),
        #         self.weight_input
        #     ),
        #     (-1, 1, self.weight_input.shape[-1])
        # )
        #
        # scores = tf.tanh(question_words + passage_word) * self.weight_score
        # scores = tf.reduce_sum(scores, -1)
        # scores = tf.nn.softmax(scores)
        scores = attention((self.attention, inputs), (self.weight_att, self.weight_input), self.weight_score)
        scores = tf.expand_dims(scores, -1)

        att = tf.reduce_sum(scores * self.attention, 1)

        # gated attention
        output = tf.concat([inputs, att], axis=1)
        gate = tf.matmul(output, self.weight_gate)
        gate = tf.sigmoid(gate)
        return gate * output

    def call(self, inputs, state):
        att = self.att(inputs)
        return super(AttentionGRUCell, self).call(att, state)


def question_pooling(question, num_units):
    with tf.variable_scope('question_pooling'):
        question_weights = tf.get_variable('W_u', shape=(question.shape[-1], num_units))
        v = tf.get_variable('v', shape=(num_units, ))

    question_pool = tf.reshape(
        tf.matmul(
            tf.reshape(question, (-1, question.shape[-1])),
            question_weights
        ),
        (question.shape[0], question.shape[1], -1)
    )

    bias = tf.get_variable('W_u_v_r', shape=(num_units, ), dtype=tf.float32)
    bias = tf.reshape(
        tf.tile(bias, [question.shape[0]]),
        (-1, 1, num_units)
    )

    scores = tf.tanh(question_pool + bias) * v
    scores = tf.reduce_sum(scores, -1)
    scores = tf.nn.softmax(scores)
    scores = tf.expand_dims(scores, -1)

    output = tf.reduce_sum(scores * question, 1)
    return output


def attention(inputs, weights, dot_vector):
    l = tf.reshape(
        tf.matmul(
            tf.reshape(inputs[0], (-1, inputs[0].shape[-1])),
            weights[0]
        ),
        (-1, inputs[0].shape[1], weights[0].shape[-1])
    )

    r = tf.reshape(
        tf.matmul(
            tf.reshape(inputs[1], (-1, inputs[1].shape[-1])),
            weights[1]
        ),
        (-1, 1, weights[1].shape[-1])
    )

    scores = tf.tanh(l + r) * dot_vector
    scores = tf.reduce_sum(scores, -1)

    return tf.nn.softmax(scores)


def pointer_network(question, passage, num_units=150):
    q_pool = question_pooling(question, num_units)

    with tf.variable_scope('pointer_network'):
        passage_weights = tf.get_variable('W_h_P', shape=(passage.shape[-1], num_units), dtype=tf.float32)
        state_weights = tf.get_variable('W_h_a', shape=(q_pool.shape[-1], num_units), dtype=tf.float32)
        v = tf.get_variable('v', shape=(num_units,), dtype=tf.float32)

    p1_logits = attention((passage, q_pool), (passage_weights, state_weights), v)

    scores = tf.expand_dims(p1_logits, -1)
    att = tf.reduce_sum(scores * passage, 1)

    rnn = GRUCell(num_units * 2)
    _, state = rnn(att, q_pool)
    p2_logits = attention((passage, state), (passage_weights, state_weights), v)

    return p1_logits, p2_logits
