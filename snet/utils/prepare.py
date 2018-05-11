import csv
import ujson
import typing
import numpy as np
from tqdm import tqdm
from collections import namedtuple
import spacy
import click
import itertools

import tensorflow as tf

from snet.utils import helpers

#from ..environment import current_env

nlp = spacy.blank("en")

example = namedtuple('example', ['passage_tokens', 'passage_chars',
                                 'question_tokens', 'question_chars',
                                 'answer_start', 'answer_end', 'answer_tokens'])

# def fix_sentences(passage):
#     r = re.compile("([\.,])([a-z])", re.I)
#     return r.sub('\1 \2', passage)
#
# tokenize = tf.keras.preprocessing.text.text_to_word_sequence
#
#
# def clean(text):
#     return " ".join(tokenize(fix_sentences(text)))

known_words = set()


def word_tokenize(sent):
    global known_words

    doc = nlp(sent)
    words = [token.text for token in doc]
    known_words |= set(words)
    return words


# def json2csv(filename, out):
#     d = ujson.loads(open(filename).read())
#
#     def shortest(lst):
#         return sorted(lst, key=len)[0]
#
#     with open(out, 'w') as f:
#         w = csv.DictWriter(f, fieldnames=['id', 'question', 'answer', 'wellFormedAnswer', 'passage'])
#         w.writeheader()
#         w.writerows((dict(id=l['query_id'],
#                           question=clean(l['query']),
#                           passage=clean(' '.join([p['passage_text'] for p in l['passages']])),
#                           answer=clean(shortest(l['answers'])),
#                           wellFormedAnswer=clean(l['wellFormedAnswers'][0]))
#                      for l in d))


# def find_answer(p_tokens, a_tokens):
#     hot_encoded = np.zeros(current_env().passage_max_length, dtype=np.float32)
#
#     def match(n_gram):
#         for i in range(len(p_tokens)):
#             if p_tokens[i:i+len(n_gram)] == n_gram:
#                 print(p_tokens[i:i+len(n_gram)], n_gram)
#                 return i
#
#     for n_gram_size in range(len(a_tokens), 0, -1):
#         n_grams = [a_tokens[i:i+n_gram_size] for i in range(0, len(a_tokens) - n_gram_size + 1)]
#         # print(n_gram_size, n_grams)
#         matches = [match(n_gram) for n_gram in n_grams]
#
#         if not matches or not all(matches):
#             continue
#
#         # print(matches, n_gram_size)
#         for m in matches:
#             for i in range(n_gram_size):
#                 hot_encoded[m + i] = 1.0
#
#         break
#
#     return hot_encoded
#
#
# def encode_answers(filename, out):
#     f = csv.DictReader(open(filename))
#     result = [find_answer(tokenize(line['passage'])[:1000], tokenize(line['answer'])) for line in f]
#
#     np.save(out, np.stack(result))



def save():
    r = csv.DictReader('data.csv')
    tokens = {w for l in r for w in word_tokenize(fix_sentences(l['passage']))}
    answer_tokens = {w for l in r for w in word_tokenize(l['answer'])}
    f = open('glove.840B.300d.txt')

    with open('glove.reduced.txt', 'w') as f2:
        for line in f:
            word = line.split(' ', 1)[0]
            if word in tokens or word in answer_tokens:
                f2.write(line)


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


def load(input_filename) -> typing.Iterator[example]:
    items = ujson.loads(open(input_filename).read())

    def shortest(lst):
        return sorted(lst, key=len)[0]

    def iter_find(find, items):
        pos = 0
        for i in items:
            pos = find(i, pos)
            yield pos, pos + len(i)

    def clean(text):
        return text.replace("''", '" ').replace("``", '" ')

    for item in tqdm(items):
        if not item['answers']:
            continue

        answer = shortest(item['answers'])

        full_passage = ' '.join([clean(p['passage_text']) for p in item['passages']])
        passage_tokens = word_tokenize(full_passage)
        passage_chars = [list(token) for token in passage_tokens]

        # spans = list(iter_find(full_passage.find, passage_tokens))

        question = clean(item['query'])
        question_tokens = word_tokenize(question)
        question_chars = [list(token) for token in question_tokens]

        answer_tokens = word_tokenize(answer)
        answer_start, answer_end = find_answer(passage_tokens, answer_tokens)

        if not answer_start or answer_end == len(passage_tokens):
            continue

        yield example(passage_tokens, passage_chars, question_tokens, question_chars,
                      answer_start, answer_end, answer_tokens)


@click.group()
def cli():
    pass


@cli.command()
@click.option('--embeddings', default='data/glove.840B.300d.txt')
@click.option('--output')
@click.argument('data-file')
def reduce_embeddings(embeddings, output, data_file):
    examples = load(data_file)
    for _ in tqdm(examples):
        pass

    counter = 0

    with open(output, 'w') as output:
        with open(embeddings, 'r') as emb:
            for line in emb:
                word, _ = line.split(" ", 1)
                if word not in known_words:
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
@click.option('--passage-words-max', default=1000, type=int)
@click.option('--question-words-max', default=30, type=int)
@click.option('--char-max', default=16, type=int)
@click.option('--word-embedding', default='data/glove.reduced.txt')
@click.option('--char-embedding', default='data/glove.840B.300d-char.txt')
@click.option('--limit', default=100)
@click.option('--tf-output')
@click.argument('data-file')
def preprocess(tf_output, passage_words_max, question_words_max, char_max,
               word_embedding, char_embedding, limit, data_file):

    examples = load(data_file)
    if limit:
        examples = itertools.islice(examples, 0, limit)

    writer = tf.python_io.TFRecordWriter(tf_output)
    word2idx, unknown = load_embeddings(word_embedding)
    char2idx, unknown = load_embeddings(char_embedding)

    def convert_words(tokens, output):
        for i, token in enumerate(tokens):
            output[i] = word2idx.get(token, unknown)

    def convert_chars(tokens, output):
        for i, token in enumerate(tokens):
            for j, char in enumerate(token[:char_max]):
                output[i, j] = char2idx.get(char, unknown)

    for example in tqdm(examples):
        if example.answer_end >= passage_words_max:
            continue

        if len(example.question_tokens) > question_words_max:
            continue

        passage_words = np.zeros((passage_words_max, ), dtype=np.int32)
        question_words = np.zeros((question_words_max, ), dtype=np.int32)

        passage_chars = np.zeros((passage_words_max, char_max), dtype=np.int32)
        question_chars = np.zeros((question_words_max, char_max), dtype=np.int32)

        answer_start = np.zeros((passage_words_max, ), dtype=np.float32)
        answer_end = np.zeros((passage_words_max, ), dtype=np.float32)

        convert_words(example.passage_tokens[:passage_words_max], passage_words)
        convert_words(example.question_tokens[:question_words_max],  question_words)

        convert_chars(example.passage_chars[:passage_words_max], passage_chars)
        convert_chars(example.question_chars[:question_words_max], question_chars)

        answer_start[example.answer_start] = 1.0
        answer_end[example.answer_end] = 1.0

        passage_char_length = np.zeros((passage_words_max, ), dtype=np.int32)
        question_char_length = np.zeros((question_words_max, ), dtype=np.int32)

        for idx, token in enumerate(example.passage_chars[:passage_words_max]):
            passage_char_length[idx] = min(len(token), char_max)
        for idx, token in enumerate(example.question_chars[:question_words_max]):
            question_char_length[idx] = min(len(token), char_max)

        passage_length = min(len(example.passage_tokens), passage_words_max)
        question_length = min(len(example.question_tokens), question_words_max)

        answer_bytes = [word.encode('utf-8') for word in example.answer_tokens]

        writer.write(
            tf.train.Example(features=tf.train.Features(feature=dict(
                passage_words=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_words.tolist())),
                question_words=tf.train.Feature(int64_list=tf.train.Int64List(value=question_words.tolist())),
                passage_chars=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_chars.reshape(-1).tolist())),
                question_chars=tf.train.Feature(int64_list=tf.train.Int64List(value=question_chars.reshape(-1).tolist())),
                passage_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[passage_length])),
                question_length=tf.train.Feature(int64_list=tf.train.Int64List(value=[question_length])),
                passage_char_length=tf.train.Feature(int64_list=tf.train.Int64List(value=passage_char_length.tolist())),
                question_char_length=tf.train.Feature(int64_list=tf.train.Int64List(value=question_char_length.tolist())),
                y1=tf.train.Feature(float_list=tf.train.FloatList(value=answer_start.tolist())),
                y2=tf.train.Feature(float_list=tf.train.FloatList(value=answer_end.tolist())),
                answer_tokens=tf.train.Feature(bytes_list=tf.train.BytesList(value=answer_bytes))
            ))).SerializeToString()
        )

    writer.close()


if __name__ == '__main__':
    cli()


