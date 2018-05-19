import tensorflow as tf

from tensorflow.python.ops.rnn_cell_impl import MultiRNNCell, GRUCell


def biGRU(input, input_length, params, layers=3):
    cell_fw = MultiRNNCell([GRUCell(params.units) for _ in range(layers)])
    cell_bw = MultiRNNCell([GRUCell(params.units) for _ in range(layers)])
    # cell_fw = DropoutWrapper(GRUCell(params.units), output_keep_prob=1 - params.dropout)
    # cell_bw = DropoutWrapper(GRUCell(params.units), output_keep_prob=1 - params.dropout)
    # input = tf.Print(input, [tf.shape(input), input_length], summarize=1000)
    output, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                                     sequence_length=input_length,
                                                     dtype=tf.float32)
    output = tf.concat(output, -1)
    return output, states


def pad_to_shape_2d(t, shape, axis=0):
    t = tf.concat([t, tf.zeros(shape)], axis=axis)[:shape[0].value, :shape[1].value]
    t.set_shape(shape)
    return t
