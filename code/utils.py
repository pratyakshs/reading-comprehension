from collections import OrderedDict
import logging
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

NUM = "NNNUMMM"


def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = np.array(list(map(float, vector.split())))

    return ret

def load_embeddings(args, helper, embed_size):
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, embed_size), dtype=np.float32)
    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(args.vocab, args.vectors).items():
        word = normalize(word)
        if word in helper.tok2id:
            embeddings[helper.tok2id[word]] = vec 
    logger.info("Initialized embeddings.")

    return embeddings
