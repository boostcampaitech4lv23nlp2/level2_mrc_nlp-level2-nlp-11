from konlpy.tag import Mecab

def konlpy_mecab_tokenizer_fn(x) :
    m = Mecab()
    return m.morphs(x)

def space_split_tokenizer_fn(x) :
    return x.split(' ')