from collections import namedtuple

_stack_env = []


class Environment(namedtuple('Environments', [
    'passage_max_length', 'question_max_length', 'answer_max_length', 'word_max_length',
    'batch_size',
    'emb_size', 'char_emb_size', 'char_vocab_size', 'vocab_size'
])):
    pass


def current_env() -> Environment:
    return _stack_env[-1]


def push_env(env: Environment):
    _stack_env.append(env)


def pop_env():
    return _stack_env.pop()

# default
push_env(Environment(passage_max_length=1000, question_max_length=50, answer_max_length=50, word_max_length=16,
                     batch_size=2, emb_size=300, char_emb_size=8, char_vocab_size=127, vocab_size=10e5))
