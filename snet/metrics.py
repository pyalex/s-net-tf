import tensorflow as tf


def f1_metric(p1, p2, answers, passage, hparams, table):
    p1 = tf.unstack(p1, hparams.batch_size)
    p2 = tf.unstack(p2, hparams.batch_size)
    answers = tf.sparse_tensor_to_dense(answers, '')
    answers = tf.unstack(answers, hparams.batch_size)
    passage = tf.unstack(passage, hparams.batch_size)

    f1 = []
    for p1_, p2_, answer, context in zip(p1, p2, answers, passage):
        answer = tf.boolean_mask(answer, tf.not_equal(answer, ''))
        prediction = table.lookup(context[p1_:p2_])

        # intersection = tf.size(tf.sets.set_intersection(tf.expand_dims(prediction, 0),
        #                                                 tf.expand_dims(answer, 0)))

        intersection = tf.py_func(lcs, [prediction, answer], tf.int64, stateful=False)
        intersection = tf.cast(intersection, tf.int32)

        precision = intersection / tf.size(prediction)
        recall = intersection / tf.size(answer)

        f1.append(tf.cond(precision + recall > 0,
                          lambda: 2 * precision * recall / (precision + recall),
                          lambda: tf.constant(.0, dtype=tf.float64)))

    f1_metrics = tf.stack(f1)
    # f1_metrics = tf.Print(f1_metrics, [f1_metrics])
    return tf.metrics.mean(f1_metrics)


def lcs(x, y):
    """
    Computes the length of the longest common subsequence (lcs) between two
    strings. The implementation below uses a DP programming algorithm and runs
    in O(nm) time where n = len(x) and m = len(y).
    Source:



    Args:
      x: collection of words
      y: collection of words

    Returns:
      Table of dictionary of coord and len lcs
    """
    n, m = len(x), len(y)
    table = dict()
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])

    return table[n, m]


def rouge_l(prediction, target, prediction_length, answer_length, params, table):
    prediction_list = tf.unstack(prediction, params.batch_size)
    target_list = tf.unstack(target, params.batch_size)
    answer_length_list = tf.unstack(answer_length, params.batch_size)
    prediction_length_list = tf.unstack(prediction_length, params.batch_size)

    rouge = []
    for p, t, pl, tl in zip(prediction_list, target_list, prediction_length_list, answer_length_list):
        intersection = tf.py_func(lcs, [p[:pl], t[:tl]], tf.int64, stateful=False)
        intersection = tf.cast(intersection, tf.int32)

        intersection = tf.Print(intersection, [table.lookup(tf.cast(p[:pl], tf.int64)),
                                               table.lookup(tf.cast(t[:tl], tf.int64)),
                                               intersection], summarize=100)

        precision = intersection / pl
        recall = intersection / tl

        rouge.append(tf.cond(precision + recall > 0,
                             lambda: 2 * precision * recall / (precision + recall),
                             lambda: tf.constant(.0, dtype=tf.float64)))

    return tf.metrics.mean(tf.stack(rouge))