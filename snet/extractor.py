import itertools
import multiprocessing
import os
from functools import partial

import click
import numpy as np
import tensorflow as tf
from tensorflow.contrib.lookup import lookup_ops
from tensorflow.contrib.seq2seq import BahdanauAttention, LuongAttention
from tensorflow.contrib.training import HParams
from tensorflow.python.client import device_lib
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from snet.helpers import biGRU, pad_to_shape_2d, ReusableBahdanauAttention, GatedAttentionWrapper
from snet.metrics import f1_metric
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


def get_devices():
    if os.environ.get('TF_DEVICES'):
        devices = os.environ.get('TF_DEVICES').split(',')
    else:
        local_device_protos = device_lib.list_local_devices()
        devices = [x.name for x in local_device_protos if x.device_type == 'GPU']

    if not devices:
        devices = ['/cpu:0']

    print('Available devices:', devices)

    return itertools.cycle(devices)


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
        answer_tokens=tf.VarLenFeature(tf.string),
        passage_ranks=tf.FixedLenFeature((hparams.passage_count,), tf.float32),
        partitions=tf.FixedLenFeature((hparams.passage_max_words,), tf.int64),
        partitions_len=tf.FixedLenFeature((hparams.passage_count,), tf.int64)
    )

    def parse(example):
        features = tf.parse_single_example(example, features_map)
        features['passage_chars'] = tf.reshape(features['passage_chars'],
                                               (hparams.passage_max_words, hparams.passage_max_chars))
        features['question_chars'] = tf.reshape(features['question_chars'],
                                                (hparams.question_max_words, hparams.question_max_chars))

        features['partitions'] = tf.cast(features['partitions'], tf.int32)

        label = (features.pop('y1'), features.pop('y2'), features.pop('passage_ranks'))
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


def encoder(word_emb, word_length, char_emb, char_length, params):
    char_emb = tf.reshape(char_emb,
                          (-1, char_emb.shape[-2], char_emb.shape[-1]))
    char_length = tf.reshape(char_length, (-1,))

    with tf.variable_scope('char_encoding'):
        _, states = biGRU(char_emb, char_length, params, layers=1)

    char_emb = tf.reshape(tf.concat(states, 1), (-1, word_emb.shape[1], 2 * params.units))

    emb = tf.concat([word_emb, char_emb], 2)
    enc, _ = biGRU(emb, word_length, params)
    return enc


def pointer_net(passage, passage_length, question_pool, params):
    attention_cell = BahdanauAttention(params.units, passage, passage_length,
                                       name="pointer_attention", probability_fn=tf.identity, score_mask_value=0)
    p1, _ = attention_cell(question_pool, None)

    context = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(p1), -1) * passage, 1)
    rnn = GRUCell(params.units * 2, name="pointer_gru")
    _, state = rnn(context, question_pool)

    p2, _ = attention_cell(state, None)
    return p1, p2


def model_fn(features, labels, mode, params, word_embeddings_np=None, char_embeddings_np=None):
    attention_fun = BahdanauAttention if params.attention == 'bahdanau' else LuongAttention
    question_words_length = features['question_length']
    passage_words_length = features['passage_length']

    devices = get_devices()

    with tf.device('/cpu:0'):
        word_embeddings_placeholder = tf.placeholder(shape=[params.vocab_size, params.emb_size], dtype=tf.float32)
        char_embeddings_placeholder = tf.placeholder(shape=[params.char_vocab_size, params.char_emb_size],
                                                     dtype=tf.float32)

        # word_embeddings = tf.create_partitioned_variables(shape=[params.vocab_size, params.emb_size],
        #                                                   slicing=[10, 1],
        #                                                   initializer=word_embeddings_placeholder,
        #                                                   trainable=False, name="word_embeddings")
        word_embeddings = tf.Variable(word_embeddings_placeholder, trainable=False, name="word_embeddings")
        char_embeddings = tf.Variable(char_embeddings_placeholder, trainable=False, name="char_embeddings")

    question_words_emb = tf.nn.embedding_lookup(word_embeddings, features['question_words'])
    question_chars_emb = tf.nn.embedding_lookup(char_embeddings, features['question_chars'])

    passage_words_emb = tf.nn.embedding_lookup(word_embeddings, features['passage_words'])
    passage_chars_emb = tf.nn.embedding_lookup(char_embeddings, features['passage_chars'])

    with tf.device(next(devices)):
        with tf.variable_scope('question_encoding'):
            question_enc = encoder(question_words_emb, question_words_length, question_chars_emb,
                                   features['question_char_length'], params)

    with tf.device(next(devices)):
        with tf.variable_scope('passage_encoding'):
            passage_enc = encoder(passage_words_emb, passage_words_length, passage_chars_emb,
                                  features['passage_char_length'], params)
        # question_enc = tf.Print(question_enc, [question_enc], summarize=1000)

        with tf.variable_scope('attention'):
            cell = GatedAttentionWrapper(
                attention_fun(params.units, question_enc, question_words_length),
                GRUCell(params.units, name="attention_gru"))

            passage_repr, _ = tf.nn.dynamic_rnn(cell, passage_enc, passage_words_length, dtype=tf.float32)

        with tf.variable_scope('pointer'):
            pool_param = tf.get_variable('pool_param', shape=(params.units,), initializer=tf.initializers.ones)
            pool_param = tf.reshape(tf.tile(pool_param, [params.batch_size]), (-1, params.units))

            question_alignments, _ = attention_fun(params.units, question_enc,
                                                   question_words_length, name="question_align")(pool_param, None)
            question_pool = tf.reduce_sum(tf.expand_dims(question_alignments, -1) * question_enc, 1)

            logits1, logits2 = pointer_net(passage_repr, passage_words_length, question_pool, params)

        with tf.variable_scope('passage_ranking'):
            W_g = Dense(params.units, activation=tf.tanh, use_bias=False)
            v_g = Dense(1, use_bias=False)

            memory_layer = Dense(
                params.units, name="memory_layer", use_bias=False, dtype=tf.float32
            )
            query_layer = Dense(
                params.units, name="query_layer", use_bias=False, dtype=tf.float32
            )
            g = []

            for i in range(params.passage_count):
                passage_mask = tf.boolean_mask(passage_repr, tf.equal(features['partitions'], i))
                passage_i = tf.split(passage_mask, features['partitions_len'][:, i])
                passage_i = [pad_to_shape_2d(p, (tf.Dimension(params.passage_max_len), p.shape[1])) for p in passage_i]
                passage_i = tf.stack(passage_i)

                passage_alignment, _ = ReusableBahdanauAttention(
                    params.units, passage_i, features['partitions_len'][:, i],
                    memory_layer=memory_layer, query_layer=query_layer, name="passage_align")(question_pool, None)

                passage_pool = tf.reduce_sum(tf.expand_dims(passage_alignment, -1) * passage_i, 1)
                g_i = v_g(W_g(tf.concat([question_pool, passage_pool], -1)))

                # g_i = tf.Print(g_i, [passage_mask, passage_i], message='is_nan_{}'.format(i), summarize=1000)
                g.append(g_i)

            g = tf.concat(g, -1)

    outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                      tf.expand_dims(tf.nn.softmax(logits2), axis=1))
    outer = tf.matrix_band_part(outer, 0, 15)
    p1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
    p2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)

    answer_start, answer_end, passage_rank = labels

    loss1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits1,
                                                       labels=tf.stop_gradient(answer_start))
    loss2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2,
                                                       labels=tf.stop_gradient(answer_end))

    loss3 = tf.nn.softmax_cross_entropy_with_logits_v2(logits=g, labels=tf.stop_gradient(passage_rank))

    loss1 = tf.Print(loss1, [tf.argmax(answer_start, -1), tf.argmax(answer_end, -1),
                             tf.reduce_mean(loss1), tf.reduce_mean(loss2), tf.reduce_mean(loss3)], message="loss")

    loss = params.r * tf.reduce_mean(loss1 + loss2) + (1 - params.r) * tf.reduce_mean(loss3)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        global_step = tf.train.get_or_create_global_step()

        grads = optimizer.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, params.grad_clip)
        train_op = optimizer.apply_gradients(zip(capped_grads, variables), global_step=global_step)

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op,
            scaffold=tf.train.Scaffold(init_feed_dict={word_embeddings_placeholder: word_embeddings_np,
                                                       char_embeddings_placeholder: char_embeddings_np}),
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        table = lookup_ops.index_to_string_table_from_file(params.word_vocab_file,
                                                           value_column_index=0, delimiter=" ")
        return EstimatorSpec(
            mode, loss=loss, eval_metric_ops={'f1': f1_metric(p1, p2,
                                                              features['answer_tokens'],
                                                              features['passage_words'], params, table)}
        )


@click.command()
@click.option('--model-dir')
@click.option('--train-data', default='data/train.tf')
@click.option('--eval-data', default='data/dev.tf')
@click.option('--word-embeddings', default='data/glove.6B.300d.txt')
@click.option('--char-embeddings', default='data/glove.840B.300d-char.txt')
@click.option('--hparams', default='', type=str, help='Comma separated list of "name=value" pairs.')
@click.option('--log-devices', type=bool, default=False)
def main(model_dir, train_data, eval_data, word_embeddings, char_embeddings, hparams, log_devices):
    tf.logging.set_verbosity(tf.logging.INFO)

    char_embeddings_np = load_embeddings(char_embeddings)

    if os.path.isfile(word_embeddings + '.npy'):
        word_embeddings_np = np.load(word_embeddings + '.npy')
    else:
        word_embeddings_np = load_embeddings(word_embeddings)
        np.save(word_embeddings, word_embeddings_np)

    hparams_ = HParams(
        num_epochs=10,
        batch_size=16,
        max_steps=1000,
        units=150,
        layers=2,
        dropout=0.0,
        question_max_words=30,
        question_max_chars=16,
        passage_max_words=800,
        passage_max_chars=16,
        vocab_size=word_embeddings_np.shape[0],
        emb_size=300,
        char_vocab_size=char_embeddings_np.shape[0],
        char_emb_size=300,
        word_vocab_file=word_embeddings,
        passage_count=10,
        passage_max_len=120,
        r=0.8,
        grad_clip=5.0,
        attention='bahdanau'
    )
    hparams = hparams_.parse(hparams)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = log_devices
    # config.intra_op_parallelism_threads = 32
    # config.inter_op_parallelism_threads = 32

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=1,
        tf_random_seed=19830610,
        model_dir=model_dir,
        save_summary_steps=1,
        session_config=config
    )

    with tf.Session() as sess:
        test = input_fn(
            [train_data],
            hparams=hparams,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size
        )

        print(sess.run([test]))

    estimator = tf.estimator.Estimator(
        model_fn=partial(model_fn, word_embeddings_np=word_embeddings_np, char_embeddings_np=char_embeddings_np),
        params=hparams, config=run_config)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            [train_data],
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
            [eval_data],
            hparams=hparams,
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=hparams.batch_size
        ),
        # exporters=[tf.estimator.LatestExporter(
        #     name="predict",  # the name of the folder in which the model will be exported to under export
        #     serving_input_receiver_fn=serving_input_fn,
        #     exports_to_keep=1,
        #     as_text=True)],
        steps=100,
        throttle_secs=3600
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
