import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
import json
import os

torch.set_grad_enabled(False)
device = "cuda"

NUM_FEATURES = 16384  # SAE width (16k)


def load_model():
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res-canonical", sae_id="layer_15/width_16k/canonical", device=device
    )

    hook_point = sae.cfg.metadata.hook_name

    return model, sae, hook_point


def sample_from_model(sae, model, steering_features, coef, batch_size, prompt="", **kwargs):
    torch.manual_seed(42)

    if not steering_features:
        steering_vector = 0
    else:
        steering_vector = sae.W_dec[steering_features].sum(dim=0) * coef

    def steering_hook(resid_pre, hook):
        resid_pre += coef * steering_vector

    def hooked_generate(prompt_batch, fwd_hooks=[], **kwargs):
        with model.hooks(fwd_hooks=fwd_hooks):
            tokenized = model.to_tokens(prompt_batch)
            result = model.generate(
                stop_at_eos=False,  # avoids a bug on MPS
                input=tokenized,
                max_new_tokens=20,
                do_sample=True,
                verbose=False,
                **kwargs,
            )
        return result

    def run_generate(example_prompt):
        model.reset_hooks()
        editing_hooks = [("blocks.15.hook_resid_post", steering_hook)]
        res = hooked_generate(
            [example_prompt] * batch_size, editing_hooks, **kwargs
        )

        res_str = model.to_string(res[:, 1:])
        return res_str

    return run_generate(prompt)


def run_experiment(sae, model):
    result_json = {}

    for feature_idx in tqdm(range(NUM_FEATURES)):
        completions_1 = sample_from_model(sae, model, [feature_idx], 1, batch_size=64)
        completions_10 = sample_from_model(sae, model, [feature_idx], 10, batch_size=64)
        result_json[feature_idx] = {
            "completions_1": completions_1,
            "completions_10": completions_10
        }

    return result_json


def main():
    model, sae, hook_point = load_model()
    result_json = run_experiment(sae, model)
    os.makedirs("steered_completions", exist_ok=True)
    with open("steered_completions/results.json", "w") as f:
        json.dump(result_json, f)


if __name__ == "__main__":
    main()
