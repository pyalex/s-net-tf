import tensorflow as tf
import multiprocessing
import numpy as np

from tensorflow.contrib.seq2seq import BahdanauAttention
from tensorflow.contrib.training import HParams
from tensorflow.contrib.lookup import lookup_ops
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper, GRUCell, RNNCell, MultiRNNCell

from snet.utils import helpers


def load_embeddings(file, dim=300):
    lines = helpers.line_count(file)

    idx2vec = np.zeros((lines + 1, dim))
    # words2idx = np.empty((lines + 1,), dtype="<U16")

    with open(file) as f:
        for idx, line in enumerate(f):
            word, *emb = line.strip().split(" ")
            if word.startswith('0.'):
                continue

            idx2vec[idx, :] = np.array(emb, np.float32)
            # words2idx[idx] = word

    # For unknown
    idx2vec[lines] = np.zeros((dim,), np.float32)
    # words2idx[lines] = 'UNK'

    # return words2idx, idx2vec
    return idx2vec


char_embeddings_np = load_embeddings('data/glove.840B.300d-char.txt')
#word_embeddings_np = load_embeddings('data/glove.42B.300d.txt')
#np.save('data/glove.42B.300d.np', word_embeddings_np)

word_embeddings_np = np.load('data/glove.42B.300d.np.npy')


def input_fn(tf_files,
             hparams,
             mode=tf.estimator.ModeKeys.EVAL,
             num_epochs=1,
             batch_size=100):
    buffer_size = 2 * batch_size + 1
    features_map = dict(
        passage_words=tf.FixedLenFeature((hparams.passage_max_words,), tf.int64),
        question_words=tf.FixedLenFeature((hparams.question_max_words,), tf.int64),
        passage_chars=tf.FixedLenFeature((hparams.passage_max_words * hparams.passage_max_chars,), tf.int64),
        question_chars=tf.FixedLenFeature((hparams.question_max_words * hparams.question_max_chars,), tf.int64),
        passage_length=tf.FixedLenFeature((), tf.int64),
        question_length=tf.FixedLenFeature((), tf.int64),
        passage_char_length=tf.FixedLenFeature((hparams.passage_max_words,), tf.int64),
        question_char_length=tf.FixedLenFeature((hparams.question_max_words,), tf.int64),
        y1=tf.FixedLenFeature((hparams.passage_max_words,), tf.float32),
        y2=tf.FixedLenFeature((hparams.passage_max_words,), tf.float32),
        answer_tokens=tf.VarLenFeature(tf.string)
    )

    def parse(example):
        features = tf.parse_single_example(example, features_map)
        features['passage_chars'] = tf.reshape(features['passage_chars'],
                                               (hparams.passage_max_words, hparams.passage_max_chars))
        features['question_chars'] = tf.reshape(features['question_chars'],
                                                (hparams.question_max_words, hparams.question_max_chars))

        label = (features.pop('y1'), features.pop('y2'))
        return features, label

    ds = tf.data.TFRecordDataset(tf_files)
    ds = ds.map(parse, num_parallel_calls=multiprocessing.cpu_count())

    if mode == tf.estimator.ModeKeys.TRAIN:
        ds = ds.shuffle(buffer_size)

    ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    ds = ds.repeat(num_epochs)
    ds = ds.prefetch(buffer_size)

    iterator = ds.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, target


def biGRU(input, input_length, params, layers=3):
    cell_fw = MultiRNNCell([GRUCell(params.units) for _ in range(layers)])
    cell_bw = MultiRNNCell([GRUCell(params.units) for _ in range(layers)])
    # cell_fw = DropoutWrapper(GRUCell(params.units), output_keep_prob=1 - params.dropout)
    # cell_bw = DropoutWrapper(GRUCell(params.units), output_keep_prob=1 - params.dropout)
    # input = tf.Print(input, [tf.shape(input), input_length], summarize=1000)
    return tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, input,
                                           sequence_length=input_length,
                                           dtype=tf.float32)


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
        # context = tf.squeeze(context, [1])

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
                           dtype=attention_mechanism.dtype)

    def _compute_attention(self, input, state):
        attention = super(GatedAttentionWrapper, self)._compute_attention(input, state)
        return self._gate(attention) * attention


def encoder(word_emb, word_length, char_emb, char_length, params):
    char_emb = tf.reshape(char_emb,
                          (-1, char_emb.shape[-2], char_emb.shape[-1]))
    char_length = tf.reshape(char_length, (-1,))

    with tf.variable_scope('char_encoding'):
        _, states = biGRU(char_emb, char_length, params, layers=1)

    char_emb = tf.reshape(tf.concat(states, 1), (-1, word_emb.shape[1], 2 * params.units))

    emb = tf.concat([word_emb, char_emb], 2)
    enc, _ = biGRU(emb, word_length, params)
    return tf.concat(enc, 2)


def pointer_net(passage, passage_length, question_pool, params):
    attention_cell = BahdanauAttention(params.units, passage, passage_length,
                                       name="pointer_attention", probability_fn=tf.identity)
    p1, _ = attention_cell(question_pool, None)

    context = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(p1), -1) * passage, 1)
    rnn = GRUCell(params.units * 2, name="pointer_gru")
    _, state = rnn(context, question_pool)

    p2, _ = attention_cell(state, None)
    return p1, p2


def f1_metric(p1, p2, answers, passage, hparams, table):
    p1 = tf.unstack(p1, hparams.batch_size)
    p2 = tf.unstack(p2, hparams.batch_size)
    answers = tf.sparse_tensor_to_dense(answers, '')
    answers = tf.unstack(answers, hparams.batch_size)
    passage = tf.unstack(passage, hparams.batch_size)

    f1 = []
    for p1_, p2_, answer, context in zip(p1, p2, answers, passage):
        p1_ = tf.Print(p1_, [p1_, p1_, answer, context], message='f1', summarize=1000)
        prediction = table.lookup(context[p1_:p2_])

        prediction = tf.Print(prediction, [prediction, answer], message='prediction', summarize=1000)
        intersection = tf.size(tf.sets.set_intersection(tf.expand_dims(prediction, 0),
                                                        tf.expand_dims(answer, 0)))
        precision = intersection / tf.size(answer)
        recall = intersection / tf.size(answer)
        f1.append(tf.cond(precision + recall > 0,
                          lambda: 2 * precision * recall / (precision + recall),
                          lambda: tf.constant(.0, dtype=tf.float64)))

    f1_metrics = tf.stack(f1)
    # f1_metrics = tf.Print(f1_metrics, [f1_metrics])
    return tf.metrics.mean(f1_metrics)


def model_fn(features, labels, mode, params):
    word_embeddings_placeholder = tf.placeholder(shape=[params.vocab_size, params.emb_size], dtype=tf.float32)
    char_embeddings_placeholder = tf.placeholder(shape=[params.char_vocab_size, params.char_emb_size], dtype=tf.float32)

    question_words_length = features['question_length']
    passage_words_length = features['passage_length']

    word_embeddings = tf.create_partitioned_variables(shape=[params.vocab_size, params.emb_size],
                                                      slicing=[10, 1],
                                                      initializer=word_embeddings_placeholder,
                                                      trainable=False, name="word_embeddings")
    char_embeddings = tf.Variable(char_embeddings_placeholder, trainable=False, name="char_embeddings")

    question_words_emb = tf.nn.embedding_lookup(word_embeddings, features['question_words'], partition_strategy='div')
    question_chars_emb = tf.nn.embedding_lookup(char_embeddings, features['question_chars'])

    passage_words_emb = tf.nn.embedding_lookup(word_embeddings, features['passage_words'], partition_strategy='div')
    passage_chars_emb = tf.nn.embedding_lookup(char_embeddings, features['passage_chars'])

    with tf.variable_scope('question_encoding'):
        question_enc = encoder(question_words_emb, question_words_length, question_chars_emb,
                               features['question_char_length'], params)

    with tf.variable_scope('passage_encoding'):
        passage_enc = encoder(passage_words_emb, passage_words_length, passage_chars_emb,
                              features['passage_char_length'], params)
    # question_enc = tf.Print(question_enc, [question_enc], summarize=1000)

    with tf.variable_scope('attention'):
        cell = GatedAttentionWrapper(
            BahdanauAttention(params.units, question_enc, question_words_length),
            GRUCell(params.units, name="attention_gru"))

        passage_repr, _ = tf.nn.dynamic_rnn(cell, passage_enc, passage_words_length, dtype=tf.float32)

    # passage_repr = tf.Print(passage_repr, [passage_repr], message='repr', summarize=50)

    with tf.variable_scope('pointer'):
        pool_param = tf.get_variable('pool_param', shape=(params.units,), initializer=tf.initializers.ones)
        pool_param = tf.reshape(tf.tile(pool_param, [params.batch_size]), (-1, params.units))

        question_alignments, _ = BahdanauAttention(params.units, question_enc,
                                                   question_words_length, name="question_align")(pool_param, None)
        question_pool = tf.reduce_sum(tf.expand_dims(question_alignments, -1) * question_enc, 1)

        logits1, logits2 = pointer_net(passage_repr, passage_words_length, question_pool, params)

    outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                      tf.expand_dims(tf.nn.softmax(logits2), axis=1))
    outer = tf.matrix_band_part(outer, 0, 15)
    p1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    p2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    answer_start, answer_end = labels
    # y1 = tf.argmax(labels, axis=-1)
    # y2 = params.passage_max_words - tf.argmax(tf.reverse(labels, axis=-1))

    loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.clip_by_value(logits1, 1e-10, 1.0),
                                                       labels=tf.stop_gradient(answer_start))
    loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.clip_by_value(logits2, 1e-10, 1.0),
                                                       labels=tf.stop_gradient(answer_end))

    loss1 = tf.Print(loss1, [tf.argmax(answer_start, -1), tf.argmax(answer_end, -1),
                             tf.reduce_mean(loss1), tf.reduce_mean(loss2)], message="truth")

    loss = tf.reduce_mean(loss1 + loss2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.5)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss=loss, global_step=global_step)

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op,
            scaffold=tf.train.Scaffold(init_feed_dict={word_embeddings_placeholder: word_embeddings_np,
                                                       char_embeddings_placeholder: char_embeddings_np}),
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        table = lookup_ops.index_to_string_table_from_file(hparams.word_vocav_file,
                                                           value_column_index=0, delimiter=" ")
        return EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'f1': f1_metric(p1, p2, features['answer_tokens'],
                                                              features['passage_words'], hparams, table)}
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    hparams = HParams(
        num_epochs=10,
        batch_size=32,
        max_steps=1000,
        units=150,
        dropout=0.0,
        question_max_words=30,
        question_max_chars=16,
        passage_max_words=400,
        passage_max_chars=16,
        vocab_size=word_embeddings_np.shape[0],
        emb_size=300,
        char_vocab_size=char_embeddings_np.shape[0],
        char_emb_size=300,
        word_vocav_file='data/glove.42B.300d.txt'
    )

    model_dir = 'trained_models/{}-init2'.format('snet')
    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1,
        tf_random_seed=19830610,
        model_dir=model_dir,
        save_summary_steps=1
    )

    with tf.Session() as sess:
        test = input_fn(
            ['data/train.tf'],
            hparams=hparams,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size
        )

        print(sess.run([test]))

    estimator = tf.estimator.Estimator(model_fn=model_fn, params=hparams, config=run_config)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            ['data/train.tf'],
            hparams=hparams,
            mode=tf.estimator.ModeKeys.TRAIN,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.batch_size
        ),
        max_steps=hparams.max_steps,
        hooks=None
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            ['data/train.tf'],
            hparams=hparams,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size
        ),
        # exporters=[tf.estimator.LatestExporter(
        #     name="predict",  # the name of the folder in which the model will be exported to under export
        #     serving_input_receiver_fn=serving_input_fn,
        #     exports_to_keep=1,
        #     as_text=True)],
        steps=1,
        throttle_secs=3600
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
