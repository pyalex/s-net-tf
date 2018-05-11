import tensorflow as tf

from snet.dataset import BatchInput
from snet.environment import current_env
from snet import layers
from snet.layers import AttentionGRUCell, question_pooling, pointer_network


def encoding(words, words_count, chars, chars_count, word_embeddings, char_embeddings, scope):
    with tf.variable_scope(scope):
        word_encoding = tf.nn.embedding_lookup(word_embeddings, words)
        char_encoding = tf.nn.embedding_lookup(char_embeddings, chars)

    char_encoding = layers.biGRU(char_encoding, inputs_len=chars_count, scope=scope + '_char_encoding', return_state=True)

    full_encoding = tf.concat([word_encoding, char_encoding], axis=2)
    return layers.biGRU(full_encoding, inputs_len=words_count, scope=scope + '_full_encoding')


class SNet:
    emb_assign = None

    def __init__(self, dataset: BatchInput):
        self.ds = dataset
        self.env = current_env()

    def encode(self):
        with tf.device('/cpu:0'):
            char_embeddings = tf.Variable(tf.constant(0.0, shape=[self.env.char_vocab_size, self.env.char_emb_size]),
                                          trainable=True, name="char_embeddings")
            word_embeddings = tf.Variable(tf.constant(0.0, shape=[self.env.vocab_size, self.env.emb_size]),
                                          trainable=False, name="word_embeddings")

            word_embeddings_placeholder = tf.placeholder(tf.float32, [self.env.vocab_size, self.env.emb_size],
                                                         "word_embeddings_placeholder")
            self.emb_assign = tf.assign(word_embeddings, word_embeddings_placeholder)

        q = encoding(self.ds.question, self.ds.question_word_count,
                     self.ds.question_chars, self.ds.question_char_count,
                     word_embeddings, char_embeddings, 'question')
        p = encoding(self.ds.passage, self.ds.passage_word_count,
                     self.ds.passage_chars, self.ds.passage_char_count,
                     word_embeddings, char_embeddings, 'passage')
        return p, q

    def build_graph(self):
        passage_emb, question_emb = self.encode()

        cell = AttentionGRUCell(num_units=150, attention_input=question_emb)
        outputs, _ = tf.nn.dynamic_rnn(cell, passage_emb,
                                       sequence_length=self.ds.passage_word_count, dtype=tf.float32)

        p1, p2 = pointer_network(passage=outputs, question=question_emb)
        print(p1, p2)


if __name__ == '__main__':
    from snet.dataset import get_train_data, glove2ind, char2ind

    word_lookup = glove2ind()
    char_lookup = char2ind()

    snet = SNet(get_train_data('data/train.csv', 'data/answers.npy', word_lookup, char_lookup))
    snet.build_graph()

    print(a)
    print(b)
