import tensorflow as tf
import multiprocessing
import click

from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper, BasicDecoder,\
    GreedyEmbeddingHelper, TrainingHelper, dynamic_decode
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.layers import maxout

from snet.helpers import biGRU


def masked_concat(a, b, a_length, b_length):
    batch_size = a.get_shape()[0]
    out = []
    for n in range(batch_size):
        out.append(tf.concat([
            a[n, :a_length[n], :],
            b[n, :, :],
            a[n, a_length[n]:, :],
        ], axis=0))

    return tf.stack(out)


def input_fn(tf_files,
             hparams,
             mode=tf.estimator.ModeKeys.EVAL,
             num_epochs=1,
             batch_size=100):
    buffer_size = 2 * batch_size + 1
    features_map = dict(
        passage_words=tf.FixedLenFeature((hparams.passage_max_words,), tf.int64),
        question_words=tf.FixedLenFeature((hparams.question_max_words,), tf.int64),
        answer_words=tf.FixedLenFeature((hparams.answer_max_words, ), tf.int64),
        passage_length=tf.FixedLenFeature((), tf.int64),
        question_length=tf.FixedLenFeature((), tf.int64),
        answer_length=tf.FixedLenFeature((), tf.int64),
        answer_start=tf.FixedLenFeature((hparams.passage_max_words,), tf.float32),
        answer_end=tf.FixedLenFeature((hparams.passage_max_words,), tf.float32),
        target=tf.FixedLenFeature((hparams.answer_max_words, ), tf.int64),
    )

    def parse(example):
        features = tf.parse_single_example(example, features_map)
        features['passage_length'] = tf.cast(features['passage_length'], tf.int32)
        features['question_length'] = tf.cast(features['question_length'], tf.int32)
        features['answer_length'] = tf.cast(features['answer_length'], tf.int32)

        label = features.pop('target')
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


def model_fn(features, labels, mode, params):
    embedding_encoder = tf.get_variable('embedding_encoder', shape=(params.vocab_size, params.emb_size))

    question_emb = tf.nn.embedding_lookup(embedding_encoder, features['question_words'])
    passage_emb = tf.nn.embedding_lookup(embedding_encoder, features['passage_words'])

    question_words_length = features['question_length']
    passage_words_length = features['passage_length']

    answer_start, answer_end = features['answer_start'], features['answer_end']
    answer_start = tf.concat([tf.expand_dims(answer_start, -1)] * 50, -1)
    answer_end = tf.concat([tf.expand_dims(answer_end, -1)] * 50, -1)

    with tf.variable_scope('passage_encoding'):
        passage_enc, (_, passage_bw_state) = biGRU(tf.concat([passage_emb, answer_start, answer_end], -1),
                                                   passage_words_length, params, layers=2)
        passage_bw_state = tf.concat(passage_bw_state, -1)

    with tf.variable_scope('question_encoding'):
        question_enc, (_, question_bw_state) = biGRU(question_emb, question_words_length, params, layers=2)
        question_bw_state = tf.concat(question_bw_state, -1)

    output_enc = masked_concat(question_enc, passage_enc, question_words_length, passage_words_length)

    decoder_state_layer = Dense(params.units, activation=tf.tanh, use_bias=True, name='decoder_state_init')
    decoder_init_state = decoder_state_layer(tf.concat([passage_bw_state, question_bw_state], -1))

    attention_mechanism = BahdanauAttention(
        params.units, output_enc,
        memory_sequence_length=passage_words_length + question_words_length)

    decoder_cell = AttentionWrapper(
        GRUCell(params.units), attention_mechanism,
        attention_layer_size=params.units, initial_cell_state=decoder_init_state)

    if mode == tf.estimator.ModeKeys.TRAIN:
        answer_emb = tf.nn.embedding_lookup(embedding_encoder, features['answer_words'])
        helper = TrainingHelper(answer_emb, features['answer_length'])
    else:
        helper = GreedyEmbeddingHelper(
            embedding_encoder,
            tf.fill([params.batch_size], params.tgt_sos_id), params.tgt_eos_id)

    projection_layer = Dense(params.vocab_size, use_bias=False)

    decoder = BasicDecoder(
        decoder_cell, helper, decoder_cell.zero_state(params.batch_size, tf.float32),
        output_layer=projection_layer)

    outputs, _, outputs_length = dynamic_decode(decoder, maximum_iterations=params.answer_max_words)
    logits = outputs.rnn_output

    logits = tf.Print(logits, [outputs.sample_id, labels], summarize=1000)

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.stop_gradient(labels[:, :tf.reduce_max(outputs_length)])

        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        target_weights = tf.sequence_mask(outputs_length, tf.reduce_max(outputs_length), dtype=logits.dtype)
        loss = tf.reduce_sum(crossent * target_weights) / params.batch_size

        optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)
        global_step = tf.train.get_or_create_global_step()

        grads = optimizer.compute_gradients(loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, params.grad_clip)
        train_op = optimizer.apply_gradients(zip(capped_grads, variables), global_step=global_step)

        return EstimatorSpec(
            mode, loss=loss, train_op=train_op,
        )


@click.command()
@click.option('--model-dir')
@click.option('--train-data', default='data/train_synthesis.tf')
@click.option('--eval-data', default='data/train_synthesis.tf')
@click.option('--hparams', default='', type=str)
def main(model_dir, train_data, eval_data, hparams):
    tf.logging.set_verbosity(tf.logging.INFO)

    hparams_override = hparams
    hparams = HParams(
        num_epochs=10,
        batch_size=16,
        max_steps=10000,
        units=150,
        dropout=0.0,
        question_max_words=30,
        passage_max_words=400,
        answer_max_words=50,
        vocab_size=30000,
        emb_size=300,
        r=0.8,
        cudnn=False,
        grad_clip=5.0,
        tgt_sos_id=1,
        tgt_eos_id=2
    )
    hparams.parse(hparams_override)

    config = tf.ConfigProto()
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

    estimator = tf.estimator.Estimator(model_fn=model_fn, params=hparams, config=run_config)

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
        steps=1,
        throttle_secs=3600
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()
