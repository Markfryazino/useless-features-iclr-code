import argparse
import numpy as np
import json
import os
from tqdm import tqdm, trange

from src.sae.analyze import SAEFeatureEffectAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="SAE Feature Effect Analyzer")
    
    parser.add_argument("--model-name", type=str, default="gemma-2-2b",
                       help="Model name (default: gemma-2-2b)")
    parser.add_argument("--sae-layer", type=int, default=15,
                       help="SAE layer number (default: 15)")
    parser.add_argument("--sequence-data-file", type=str, default="neuronpedia_sequences.json",
                       help="Sequence data file path (default: neuronpedia_sequences.json)")
    parser.add_argument("--max-depth", type=int, default=10,
                       help="Maximum depth (default: 10)")
    parser.add_argument("--sae-release", type=str, default="gemma-scope-2b-pt-res-canonical",
                       help="SAE release name (default: gemma-scope-2b-pt-res-canonical)")
    parser.add_argument("--sae-width", type=str, default="16k",
                       help="SAE width (default: 16k)")
    parser.add_argument("--sae-type", type=str, default="canonical",
                       help="SAE type (default: canonical)")
    parser.add_argument("--output-folder", type=str, default="sae_analysis_output",
                       help="Output folder name (default: sae_analysis_output)")
    parser.add_argument("--min-feature-idx", type=int, default=None,
                       help="Minimum feature index to analyze (inclusive, default: None)")
    parser.add_argument("--max-feature-idx", type=int, default=None,
                       help="Maximum feature index to analyze (inclusive, default: None)")
    return parser.parse_args()


def analyze_feature(data, analyzer, feature_idx, max_depth):
    results = []
    diff = np.zeros(max_depth)
    for example in data[str(feature_idx)]:
        if example["max_activation_idx"] + max_depth >= len(example["tokens"]):
            continue

        result = analyzer.analyze_sequence(
            feature_idx=feature_idx,
            token_ids=analyzer.model.tokenizer.convert_tokens_to_ids([analyzer.model.tokenizer.bos_token] + example["tokens"]),
            ablation_pos=example["max_activation_idx"] + 1
        )
        results.append(result)
        diff += np.array([pos_dict.get("kl_divergence") for pos_dict in result])
    return (diff / len(results)).tolist(), results


if __name__ == "__main__":
    args = parse_args()
    
    config = {
        "model_name": args.model_name,
        "sae_layer": args.sae_layer,
        "sequence_data_file": args.sequence_data_file,
        "max_depth": args.max_depth,
        "sae_release": args.sae_release,
        "sae_width": args.sae_width,
        "sae_type": args.sae_type,
        "output_folder": args.output_folder
    }

    analyzer = SAEFeatureEffectAnalyzer(config)

    with open(config["sequence_data_file"], "r") as f:
        data = json.load(f)

    full_results = {}

    os.makedirs(config["output_folder"], exist_ok=True)

    i = 0
    batch = {}

    all_features = np.arange(len(data))
    rng = np.random.default_rng(42)
    rng.shuffle(all_features)

    min_feature = args.min_feature_idx or 0
    max_feature = args.max_feature_idx or len(data) - 1
    all_features = all_features[min_feature: max_feature + 1]

    for feature_idx in tqdm(all_features):
        feature_idx = int(feature_idx)
        shape, results = analyze_feature(data, analyzer, feature_idx, config["max_depth"])
        batch[feature_idx] = [shape, results]
        i += 1
        if i % 100 == 0:
            output_path = os.path.join(config["output_folder"], f"sae_analysis_{i}.json")
            with open(output_path, "w") as f_out:
                json.dump(batch, f_out)
            batch = {}
    # Save any remaining results at the end
    if batch:
        output_path = os.path.join(config["output_folder"], f"sae_analysis_{i}.json")
        with open(output_path, "w") as f_out:
            json.dump(batch, f_out)
    
    print("That's all!")
