import tensorflow as tf

from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell, GRUCell, RNNCell, DropoutWrapper


def biGRU(input, input_length, params, layers=None):
    cell_fw = MultiRNNCell([DropoutWrapper(GRUCell(params.units), output_keep_prob=1.0 - params.dropout)
                            for _ in range(layers or params.layers)])
    cell_bw = MultiRNNCell([DropoutWrapper(GRUCell(params.units), output_keep_prob=1.0 - params.dropout)
                            for _ in range(layers or params.layers)])

    output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                                     sequence_length=input_length,
                                                     dtype=tf.float32)
    output = tf.concat(output, -1)
    return output, states


def pad_to_shape_2d(t, shape, axis=0):
    t = tf.concat([t, tf.zeros(shape)], axis=axis)[:shape[0].value, :shape[1].value]
    t.set_shape(shape)
    return t


class ReusableBahdanauAttention(BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 query_layer=None,
                 memory_layer=None,
                 name="BahdanauAttention"):
        if probability_fn is None:
            probability_fn = tf.nn.softmax

        wrapped_probability_fn = lambda score, _: probability_fn(score)

        super(BahdanauAttention, self).__init__(
            query_layer=query_layer,
            memory_layer=memory_layer,
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name


class AttentionWrapper(RNNCell):
    def __init__(self, attention_mechanism, cell: RNNCell):
        self._attention_mechanism = attention_mechanism
        self._cell = cell

        super(RNNCell, self).__init__()

    @property
    def output_size(self):
        return self._cell.output_size

    @property
    def state_size(self):
        return self._cell.state_size

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    def _compute_attention(self, input, state):
        alignments, _ = self._attention_mechanism(input, state)
        expanded_alignments = tf.expand_dims(alignments, -1)

        context = tf.reduce_sum(expanded_alignments * self._attention_mechanism.values, 1)
        return tf.concat([input, context], axis=1)

    def call(self, inputs, state):  # pylint: disable=signature-differs
        attention = self._compute_attention(inputs, state)
        # attention = tf.Print(attention, [attention, inputs, state], summarize=100)
        return self._cell(attention, state)


class GatedAttentionWrapper(AttentionWrapper):
    def __init__(self, attention_mechanism, *args, **kwargs):
        super(GatedAttentionWrapper, self).__init__(attention_mechanism, *args, **kwargs)

        self._gate = Dense(activation=tf.sigmoid,
                           units=4 * attention_mechanism._num_units,
                           use_bias=False,
                           dtype=attention_mechanism.dtype,
                           kernel_initializer=tf.initializers.truncated_normal)

    def _compute_attention(self, input, state):
        attention = super(GatedAttentionWrapper, self)._compute_attention(input, state)
        gate = self._gate(attention)
        gate = tf.nn.dropout(gate, 0.8)
        return gate * attention