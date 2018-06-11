import tensorflow as tf
import numpy as np

from snet import metrics

extractor_dir = 'trained_models/snet-ext-50-luong-0.8/export/predict/1528652135'
synthesis_dir = 'trained_models/snet-syn7/export/predict/1528657273'

tfrecord = 'data/dev.new.tf'
word_emb = 'data/glove.42B.reduced.all.300d.txt'
char_emb = 'data/glove.840B.300d-char.txt'

# nlp = spacy.blank("en")
#
# contexts = [
#     "context"
# ]
#
# questions = [
#     "question"
# ]
#
# context_size = 800
# question_size = 30
# word_size = 16

extractor_fn = tf.contrib.predictor.from_saved_model(
    export_dir=extractor_dir,
    signature_def_key="prediction"
)

synthesis_fn = tf.contrib.predictor.from_saved_model(
    export_dir=synthesis_dir,
    signature_def_key="prediction"
)


def pad(lst, size, val=""):
    return lst + [val] * (size - len(lst))


with open(word_emb) as f:
    idx2word = [line.split()[0] for line in f if line.strip()]
    idx2word[0] = ""
    idx2word.append("")

with open(char_emb) as f:
    idx2char = [line.split()[0] for line in f if line.strip()]
    idx2char.append("")


def main(passage_length=800):
    # contexts_tokens = np.array([pad([token.text for token in nlp(c)], context_size) for c in contexts])
    # context_chars = np.array([[pad(list(token), word_size) for token in c] for c in contexts_tokens])
    #
    # question_tokens = np.array([pad([token.text for token in nlp(c)], question_size) for c in questions])
    # questions_chars = np.array([[pad(list(token), word_size) for token in q] for q in question_tokens])

    results = []

    for record in tf.python_io.tf_record_iterator(tfrecord):
        example = tf.train.Example()
        example.ParseFromString(record)

        features = lambda name: example.features.feature[name].int64_list.value
        truth = [bytes.decode('utf-8') for bytes in example.features.feature['answer_tokens'].bytes_list.value]

        output = extractor_fn(
            {
                'context': np.array([idx2word[idx] for idx in features('passage_words')]).reshape(-1, passage_length),
                'question': np.array([idx2word[idx] for idx in features('question_words')]).reshape(-1, 30),
                'context_chars': np.array([idx2char[idx] for idx in features('passage_chars')]).reshape(-1, passage_length, 16),
                'question_chars': np.array([idx2char[idx] for idx in features('question_chars')]).reshape(-1, 30, 16)
            }
        )

        prediction = features('passage_words')[output['start'][0]:output['end'][0] + 1]
        prediction = [idx2word[idx] for idx in prediction]

        common = metrics.lcs(prediction, truth)

        precision = common / len(prediction)
        recall = common / len(truth)
        rouge_l = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(rouge_l)

        answer_start = np.zeros(2 * passage_length)
        answer_end = np.zeros(2 * passage_length)

        answer_start[output['start'][0]] = 1.0
        answer_end[output['end'][0]] = 1.0

        prediction = synthesis_fn(
            {
                'context': np.array([idx2word[idx] for idx in features('passage_words')] * 2).reshape(-1, passage_length),
                'question': np.array([idx2word[idx] for idx in features('question_words')] * 2).reshape(-1, 30),
                'answer_start': answer_start.reshape(-1, passage_length),
                'answer_end': answer_end.reshape(-1, passage_length)
            }
        )['answer'][0].tolist()

        prediction = [b.decode('utf-8') for b in prediction]
        prediction = prediction[:prediction.index('</s>')]

        common = metrics.lcs(prediction, truth)

        precision = common / len(prediction)
        recall = common / len(truth)

        results.append(2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
        print('rouge-l', np.mean(results))

if __name__ == '__main__':
    main()
