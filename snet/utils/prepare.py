import ujson
import typing
import numpy as np
from tqdm import tqdm
from collections import namedtuple, Counter
import spacy
import click
import itertools

import tensorflow as tf

from snet.utils import helpers

nlp = spacy.blank("en")

example = namedtuple('example', ['passage_tokens', 'passage_chars',
                                 'question_tokens', 'question_chars',
                                 'answer_start', 'answer_end', 'answer_tokens',
                                 'partitions', 'partitions_len',
                                 'passage_ranks'])

_known_words = Counter()


def word_tokenize(sent):
    doc = nlp(sent)
    words = [token.text for token in doc]
    for word in words:
        _known_words[word] += 1
    return words


def find_answer(passage, answer):
    passage = [t.lower() for t in passage]
    answer = [t.lower() for t in answer]

    m, n = len(passage), len(answer)

    L = np.zeros((m + 1, n + 1), dtype=np.int64)

    # Following steps build L[m+1][n+1] in bottom up fashion. Note
    # that L[i][j] contains length of LCS of X[0..i-1] and Y[0..j-1]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif passage[i - 1] == answer[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    # Following code is used to print LCS
    index = L[m][n]

    # initialized answer start and end index
    answer_start = 0
    answer_end = m
    answer_end_match = False

    # Create a character array to store the lcs string
    lcs = [""] * (int(index) + 1)
    lcs[index] = "\0"

    # Start from the right-most-bottom-most corner and
    # one by one store characters in lcs[]
    i = m
    j = n
    while i > 0 and j > 0:

        # If current character in X[] and Y are same, then
        # current character is part of LCS
        if passage[i - 1] == answer[j - 1]:
            lcs[index - 1] = passage[i - 1]
            i -= 1
            j -= 1
            index -= 1
            if not answer_end_match:
                answer_end = i
                answer_end_match = True
            answer_start = i

        # If not same, then find the larger of two and
        # go in the direction of larger value
        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    # print "LCS of " + X + " and " + Y + " is " + "".join(lcs)
    # if answer_start == answer_end:
    #   answer_end += 1
    return answer_start, answer_end + 1


def load(input_filename, passage_words_max=800, only_selected=False) -> typing.Iterator[example]:
    f = open(input_filename, 'r')
    items = ujson.loads(f.readline())

    def shortest(lst: typing.List[str]) -> str:
        return sorted(lst, key=len)[0]

    def clean(text):
        return text.replace("''", '" ').replace("``", '" ').lower().strip(' .').lstrip('()_')

    for key, passages in items['passages'].items():
        answers = items['answers'][key]
        if not answers or 'no answer' in answers[0].lower():
            continue

        partitions = []
        partitions_len = []
        passage_tokens = []

        if only_selected:
            passages = [p for p in passages if p['is_selected']]

        for idx, p in enumerate(passages):
            text = clean(p['passage_text'])
            tokens = word_tokenize(text)

            tokens = tokens[:passage_words_max - len(passage_tokens)]

            passage_tokens.extend(tokens)
            partitions.extend([idx] * len(tokens))
            partitions_len.append(len(tokens))

        if not only_selected and (not all(partitions_len) or len(partitions_len) != 10):
            continue

        passage_ranks = [p['is_selected'] for p in passages]
        passage_chars = [list(token) for token in passage_tokens]

        question = clean(items['query'][key])
        question_tokens = word_tokenize(question)
        question_chars = [list(token) for token in question_tokens]

        answer_candidates = []

        for answer in answers:
            answer = clean(answer)
            tokens = word_tokenize(answer)
            start, end = find_answer(passage_tokens, tokens)

            if end - start > 2 * len(tokens):
                continue

            answer_candidates.append((tokens, (start, end)))

        if not answer_candidates:
            continue

        answer_tokens, (answer_start, answer_end) = sorted(answer_candidates, key=lambda x: len(x[0]))[0]

        yield example(passage_tokens, passage_chars, question_tokens, question_chars,
                      answer_start, answer_end, answer_tokens, partitions, partitions_len, passage_ranks)


@click.group()
def cli():
    pass


UNK = 'unk'
SOS = '<s>'
EOS = '</s>'


@cli.command()
@click.option('--embeddings', default='data/glove.6B.300d.txt')
@click.option('--limit', default=None, type=int)
@click.option('--most-common', default=30000, type=int)
@click.option('--most-common-output', default=None, type=str)
@click.option('--reduce_output', default=None, type=str)
@click.argument('data-file')
def vocab(embeddings, limit, most_common, most_common_output, reduce_output, data_file):
    examples = load(data_file)
    if limit:
        examples = itertools.islice(examples, 0, limit)

    for _ in tqdm(examples):
        pass

    words = [UNK, SOS, EOS] + [word for (word, _) in _known_words.most_common(most_common - 3)]

    if most_common_output:
        with open(most_common_output, 'w') as f:
            for word in words:
                f.write(word + '\n')

    if reduce_output:
        counter = 0

        with open(reduce_output, 'w') as output:
            with open(embeddings, 'r') as emb:
                for line in emb:
                    word, _ = line.split(" ", 1)
                    if word not in _known_words:
                        continue

                    output.write(line)
                    counter += 1

        print('reduced to', counter, 'lines')


def load_embeddings(filename):
    idxs = {}

    for idx, line in enumerate(open(filename)):
        word, _ = line.split(" ", 1)
        idxs[word] = idx

    unknown = len(idxs)

    return idxs, unknown


@cli.command()
@click.option('--passage-words-max', default=800, type=int)
@click.option('--question-words-max', default=30, type=int)
@click.option('--passage-count', default=10, type=int)
@click.option('--char-max', default=16, type=int)
@click.option('--word-embedding', default='data/glove.6B.300d.txt')
@click.option('--char-embedding', default='data/glove.840B.300d-char.txt')
@click.option('--limit', default=None, type=int)
@click.option('--tf-output')
@click.argument('data-file')
def extraction(tf_output, passage_words_max, question_words_max, passage_count,
               char_max, word_embedding, char_embedding, limit, data_file):
    examples = load(data_file, passage_words_max=passage_words_max)
    if limit:
        examples = itertools.islice(examples, 0, limit)

    writer = tf.python_io.TFRecordWriter(tf_output)
    word2idx, unknown_word = load_embeddings(word_embedding)
    char2idx, unknown_char = load_embeddings(char_embedding)

    def convert_words(tokens, output):
        for i, token in enumerate(tokens):
            output[i] = word2idx.get(token, unknown_word)

    def convert_chars(tokens, output):
        for i, token in enumerate(tokens):
            for j, char in enumerate(token[:char_max]):
                output[i, j] = char2idx.get(char, unknown_char)

    for example in tqdm(examples):
        if example.answer_end >= passage_words_max:
            continue

        if len(example.question_tokens) > question_words_max:
            continue

        passage_words = np.zeros((passage_words_max,), dtype=np.int32)
        question_words = np.zeros((question_words_max,), dtype=np.int32)

        passage_chars = np.zeros((passage_words_max, char_max), dtype=np.int32)
        question_chars = np.zeros((question_words_max, char_max), dtype=np.int32)

        answer_start = np.zeros((passage_words_max,), dtype=np.float32)
        answer_end = np.zeros((passage_words_max,), dtype=np.float32)

        convert_words(example.passage_tokens[:passage_words_max], passage_words)
        convert_words(example.question_tokens[:question_words_max], question_words)

        convert_chars(example.passage_chars[:passage_words_max], passage_chars)
        convert_chars(example.question_chars[:question_words_max], question_chars)

        answer_start[example.answer_start] = 1.0
        answer_end[example.answer_end] = 1.0

        passage_char_length = np.zeros((passage_words_max,), dtype=np.int32)
        question_char_length = np.zeros((question_words_max,), dtype=np.int32)

        for idx, token in enumerate(example.passage_chars[:passage_words_max]):
            passage_char_length[idx] = min(len(token), char_max)
        for idx, token in enumerate(example.question_chars[:question_words_max]):
            question_char_length[idx] = min(len(token), char_max)

        passage_length = min(len(example.passage_tokens), passage_words_max)
        question_length = min(len(example.question_tokens), question_words_max)

        answer_bytes = [word.encode('utf-8') for word in example.answer_tokens]

        passage_ranks = np.pad(example.passage_ranks,
                               (0, passage_count - len(example.passage_ranks)),
                               mode='constant', constant_values=0).astype(np.float32)

        partitions = np.pad(example.partitions[:passage_words_max],
                            (0, passage_words_max - len(example.partitions[:passage_words_max])),
                            mode='constant', constant_values=10)
        partitions_len = np.pad(example.partitions_len,
                                (0, passage_count - len(example.partitions_len)),
                                mode='constant', constant_values=0)

        writer.write(
            tf.train.Example(features=tf.train.Features(feature=dict(
                passage_words=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_words.tolist())),
                question_words=tf.train.Feature(int64_list=tf.train.Int64List(value=question_words.tolist())),
                passage_chars=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_chars.reshape(-1).tolist())),
                question_chars=tf.train.Feature(
                    int64_list=tf.train.Int64List(value=question_chars.reshape(-1).tolist())),
                passage_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[passage_length])),
                question_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[question_length])),
                passage_char_length=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_char_length.tolist())),
                question_char_length=tf.train.Feature(
                    int64_list=tf.train.Int64List(value=question_char_length.tolist())),
                y1=tf.train.Feature(float_list=tf.train.FloatList(value=answer_start.tolist())),
                y2=tf.train.Feature(float_list=tf.train.FloatList(value=answer_end.tolist())),
                passage_ranks=tf.train.Feature(float_list=tf.train.FloatList(value=passage_ranks.tolist())),
                partitions=tf.train.Feature(int64_list=tf.train.Int64List(value=partitions.tolist())),
                partitions_len=tf.train.Feature(int64_list=tf.train.Int64List(value=partitions_len.tolist())),
                answer_tokens=tf.train.Feature(bytes_list=tf.train.BytesList(value=answer_bytes))
            ))).SerializeToString()
        )

    writer.close()


@cli.command()
@click.option('--passage-words-max', default=400, type=int)
@click.option('--question-words-max', default=30, type=int)
@click.option('--answer-words-max', default=50, type=int)
@click.option('--vocab', default='data/vocab.txt', type=str)
@click.option('--limit', default=None, type=int)
@click.option('--tf-output')
@click.argument('data-file')
def synthesis(passage_words_max, question_words_max, answer_words_max, vocab,
              limit, tf_output, data_file):
    examples = load(data_file, passage_words_max=passage_words_max, only_selected=True)
    if limit:
        examples = itertools.islice(examples, 0, limit)

    writer = tf.python_io.TFRecordWriter(tf_output)

    with open(vocab) as f:
        vocab_dict = {word.strip(): idx for idx, word in enumerate(f)}

    for example in tqdm(examples):
        if len(example.answer_tokens) >= answer_words_max - 1:
            print(1)
            continue

        if example.answer_end >= passage_words_max:
            print(2)
            continue

        if len(example.question_tokens) > question_words_max:
            print(3)
            continue

        passage_words = np.zeros((passage_words_max,), dtype=np.int32)
        question_words = np.zeros((question_words_max,), dtype=np.int32)
        answer_words = np.zeros((answer_words_max, ), dtype=np.int32)
        target = np.zeros((answer_words_max, ), dtype=np.int32)
        answer_start = np.zeros((passage_words_max,), dtype=np.float32)
        answer_end = np.zeros((passage_words_max,), dtype=np.float32)

        for idx, token in enumerate(example.passage_tokens[:passage_words_max]):
            passage_words[idx] = vocab_dict.get(token, 0)

        for idx, token in enumerate(example.question_tokens[:question_words_max]):
            question_words[idx] = vocab_dict.get(token, 0)

        answer_words[0] = vocab_dict.get(SOS)

        for idx, token in enumerate(example.answer_tokens[:answer_words_max - 1]):
            answer_words[idx + 1] = vocab_dict.get(token, vocab_dict[UNK])
            target[idx] = vocab_dict.get(token, vocab_dict[UNK])

        target[idx + 1] = vocab_dict.get(EOS)

        answer_start[example.answer_start] = 1.0
        answer_end[example.answer_end] = 1.0

        passage_length = min(len(example.passage_tokens), passage_words_max)
        question_length = min(len(example.question_tokens), question_words_max)
        answer_length = min(len(example.answer_tokens) + 1, answer_words_max)

        writer.write(
            tf.train.Example(features=tf.train.Features(feature=dict(
                passage_words=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_words.tolist())),
                question_words=tf.train.Feature(int64_list=tf.train.Int64List(value=question_words.tolist())),
                answer_words=tf.train.Feature(int64_list=tf.train.Int64List(value=answer_words.tolist())),
                passage_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[passage_length])),
                question_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[question_length])),
                answer_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[answer_length])),
                answer_start=tf.train.Feature(float_list=tf.train.FloatList(value=answer_start.tolist())),
                answer_end=tf.train.Feature(float_list=tf.train.FloatList(value=answer_end.tolist())),
                target=tf.train.Feature(int64_list=tf.train.Int64List(value=target))
            ))).SerializeToString()
        )

    writer.close()


if __name__ == '__main__':
    cli()
