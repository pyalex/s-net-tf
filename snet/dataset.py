from collections import namedtuple

import tensorflow as tf
import numpy as np
import multiprocessing

from functools import partial

from . import environment
from tensorflow.python.ops import lookup_ops

# CSV Format
# 'id', 'question', 'answer', 'wellFormedAnswer', 'passage'


class BatchInput(namedtuple('BatchInput', ['question', 'question_chars', 'question_word_count', 'question_char_count',
                                           'passage', 'passage_chars', 'passage_word_count', 'passage_char_count',
                                           'answer', 'init'])):
    pass


def split_line(line):
    # Split the line into words
    line = tf.expand_dims(line, axis=0)
    line = tf.string_split(line, delimiter=' ')

    # Loop over the resulting words, split them into characters, and stack them back together
    def body(index, words):
        next_word = tf.sparse_slice(line, start=tf.to_int64(index), size=[1, 1]).values
        next_word = tf.string_split(next_word, delimiter='')
        words = tf.sparse_concat(axis=0, sp_inputs=[words, next_word], expand_nonconcat_dim=True)
        return index+[0, 1], words

    def condition(index, words):
        return tf.less(index[1], tf.size(line))

    i0 = tf.constant([0, 1])
    first_word = tf.string_split(tf.sparse_slice(line, [0, 0], [1, 1]).values, delimiter='')
    _, line = tf.while_loop(condition, body, loop_vars=[i0, first_word], back_prop=False)

    # Convert to dense
    return tf.sparse_tensor_to_dense(line, default_value=' ')


def word2ind(path='data/glove.reduced.txt'):
    return lookup_ops.index_table_from_tensor(
        tf.constant([l.split(" ")[0] for l in open(path)])
    )


def char2ind(path='data/glove.840B.300d-char.txt'):
    return lookup_ops.index_table_from_tensor(
        tf.constant([l.split(" ")[0] for l in open(path)])
    )


def get_train_data(filename, answer_filename, word_lookup, char_lookup):
    source_ds = tf.data.TextLineDataset(tf.constant([filename]))
    source_ds = source_ds.skip(1)  # skip csv-header
    source_ds = source_ds.map(partial(tf.decode_csv, record_defaults=[[0], [""], [""], [""], [""]]),
                              num_parallel_calls=multiprocessing.cpu_count())

    env = environment.current_env()

    source_ds = source_ds.map(
        lambda _1, question, _2, _3, passage:
            (tf.string_split([question]).values[:env.question_max_length],
             split_line(question)[:env.question_max_length, :env.word_max_length],
             tf.string_split([passage]).values[:env.passage_max_length],
             split_line(passage)[:env.passage_max_length, :env.word_max_length])
    )

    source_ds = source_ds.map(
        lambda question, question_c, passage, passage_c: (
            word_lookup.lookup(question), char_lookup.lookup(question_c),
            word_lookup.lookup(passage), char_lookup.lookup(passage_c),
        )
    )

    def char_size(t):
        return tf.size(tf.boolean_mask(t, tf.greater(t, 32)))

    source_ds = source_ds.map(
        lambda question, question_c, passage, passage_c:
        (question, question_c, passage, passage_c,
         tf.size(question),
         tf.map_fn(char_size, tf.cast(question_c, dtype=tf.int32)),
         tf.size(passage),
         tf.map_fn(char_size, tf.cast(passage_c, dtype=tf.int32))
         )
    )
    answer_idx = np.load(answer_filename)
    target_ds = tf.data.Dataset.from_tensor_slices((answer_idx, ))

    ds = tf.data.Dataset.zip((source_ds, target_ds))

    batched_ds = ds.padded_batch(
        env.batch_size, padded_shapes=(
            (tf.TensorShape([env.question_max_length]),
             tf.TensorShape([env.question_max_length, env.word_max_length]),
             tf.TensorShape([env.passage_max_length]),
             tf.TensorShape([env.passage_max_length, env.word_max_length]),
             tf.TensorShape([]),
             tf.TensorShape([env.question_max_length]),
             tf.TensorShape([]),
             tf.TensorShape([env.passage_max_length])),
            (tf.TensorShape([None]), )
        )
    )

    batched_iterator = batched_ds.make_initializable_iterator()
    (q, q_c, p, p_c, q_wc, q_cc, p_wc, p_cc), a = batched_iterator.get_next()
    return BatchInput(question=q, question_chars=q_c,
                      question_word_count=q_wc, question_char_count=q_cc,
                      passage=p, passage_chars=p_c,
                      passage_word_count=p_wc, passage_char_count=p_cc,
                      answer=a, init=batched_iterator.initializer)


if __name__ == '__main__':
    sess = tf.InteractiveSession()

    word_lookup = glove2ind()
    char_lookup = char2ind()

    b = get_train_data('data/train.csv', 'data/answers.npy', word_lookup, char_lookup)

    sess.run(tf.tables_initializer())
    sess.run(b.init)

    print(sess.run(b.question))
    print(sess.run(b.question_chars))
    print(sess.run(b.question_word_count))
    print(sess.run(b.question_char_count))