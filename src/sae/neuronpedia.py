from dataclasses import dataclass
from typing import List
from editdistance import distance
import json
import os
from tqdm import tqdm
from collections import defaultdict


@dataclass
class NeuronpediaSequence:
    text: str
    tokens: List[str]
    max_activation_idx: int
    max_activation_value: float

    def to_json(self):
        return {
            "text": self.text,
            "tokens": self.tokens,
            "max_activation_idx": self.max_activation_idx,
            "max_activation_value": self.max_activation_value,
        }

    @staticmethod
    def from_json(data):
        return NeuronpediaSequence(
            text=data["text"],
            tokens=data["tokens"],
            max_activation_idx=data["max_activation_idx"],
            max_activation_value=data["max_activation_value"],
        )


def add_line(line, config, all_features_seqs):
    line = json.loads(line)
    feature_idx = int(line["index"])
    max_val = line["maxValue"]
    if max_val == 0:
        return
    
    max_idx = line["maxValueTokenIndex"]
    
    tokens = line["tokens"]
    tokens = tokens[:max_idx + config["analysis_depth"] + 1]

    text = "".join(tokens).replace("▁", " ")

    all_features_seqs[feature_idx].append(
        NeuronpediaSequence(
            text=text,
            tokens=tokens,
            max_activation_idx=max_idx,
            max_activation_value=max_val
        )
    )


def distance_filtering(data, editdistance_threshold=10):
    new_result = {}

    for key, seqs in tqdm(data.items(), total=len(data)):
        seqs = sorted(seqs, key=lambda x: x.max_activation_value, reverse=True)

        new_result[key] = []

        for seq in seqs:
            novel = True
            for prev_seq in new_result[key]:
                if distance(prev_seq.tokens, seq.tokens) <= editdistance_threshold:
                    novel = False
                    break
            if novel:
                new_result[key].append(seq)

    return new_result


def full_pipeline(config, path_to_data):
    result = defaultdict(list)

    files = os.listdir(path_to_data)
    for fname in tqdm(files):
        with open(os.path.join(path_to_data, fname), "r") as f:
            lines = f.readlines()

        for line in lines:
            add_line(line, config, result)

    new_result = distance_filtering(result, editdistance_threshold=config["editdistance_threshold"])

    return new_result


if __name__ == "__main__":
    config = {
        "model_name": "gemma-2-2b",
        "sae_layer": 15,
        "analysis_depth": 10,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_width": "16k",
        "sae_type": "canonical",
        "sae_id": "15-gemmascope-res-16k",
        "editdistance_threshold": 10
    }

    path_to_data = "INSERT YOURS"

    result = full_pipeline(config, path_to_data)
    result = {k: [seq.to_json() for seq in v] for k, v in result.items()}

    with open("INSERT YOURS", "w") as f:
        json.dump(result, f)