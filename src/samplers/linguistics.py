from typing import List, Tuple, Dict, Any
from collections import defaultdict
import spacy
import numpy as np


nlp = spacy.load("en_core_web_sm")


TOP_TAGS = ['NN', 'VBD', 'DT', 'PRP', 'JJ', 'NNP', 'IN', 'RB', 'VB', 'CC']
TOP_DEPS = ['nsubj', 'ROOT', 'det', 'dobj', 'pobj', 'advmod', 'prep', 'cc', 'conj', 'aux']


def token_list_to_offsets(token_list: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Convert a list of tokens to a list of character offsets.
    """
    offsets = []
    current_offset = 0
    string = ""
    for token in token_list:
        offsets.append((current_offset, current_offset + len(token)))
        current_offset += len(token)
        string += token

    string = string.replace("Ġ", " ")
    return string, offsets

def tokens_str_to_features(tokens_str: List[str]):
    string, offsets = token_list_to_offsets(tokens_str)
    doc = nlp(string)
    token_features = []
    for idx, (start, end) in enumerate(offsets):
        token_features.append(defaultdict(set))
        for spacy_token in doc.char_span(start, end, alignment_mode="expand"):
            token_features[-1]["tags"].add(spacy_token.tag_)
            token_features[-1]["pos"].add(spacy_token.pos_)
            token_features[-1]["dep"].add(spacy_token.dep_)
            token_features[-1]["text"].add(spacy_token.text)
        token_features[-1]["pos_int"] = idx
        token_features[-1]["pos_prop"] = idx / len(tokens_str)
        token_features[-1] = dict(token_features[-1])
    return token_features


def features_to_vector(features: List[Dict[str, Any]]) -> np.ndarray:
    vector = np.zeros((len(features), len(TOP_TAGS) + len(TOP_DEPS) + 2), dtype=np.float32)
    for i, features in enumerate(features):
        for tag in TOP_TAGS:
            if "tags" in features and tag in features["tags"]:
                vector[i, TOP_TAGS.index(tag)] = 1
        for dep in TOP_DEPS:
            if "dep" in features and dep in features["dep"]:
                vector[i, len(TOP_TAGS) + TOP_DEPS.index(dep)] = 1
        vector[i, -2] = features["pos_int"]
        vector[i, -1] = features["pos_prop"]

    return vector
