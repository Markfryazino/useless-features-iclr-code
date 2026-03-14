import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer
from sae_lens import SAE
from typing import List, Dict, Optional, Tuple
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime
import argparse

from collections import OrderedDict
import torch.func as Ffunc


class SAEFeatureEffectAnalyzer:
    """Analyze the causal effects of SAE features on model predictions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing analyzer for {config['model_name']}, layer {config['sae_layer']}")
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = HookedTransformer.from_pretrained(
            config['model_name'],
            device=self.device,
            torch_dtype=torch.float32
        )
        self._ensure_pad_token()
        
        # Setup SAE
        self.layer = config['sae_layer']
        self.hook_point = f"blocks.{self.layer}.hook_resid_post"
        self._load_sae()
        
    def _ensure_pad_token(self):
        """Ensure tokenizer has pad token"""
        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id
    
    def _load_sae(self):
        """Load SAE model"""
        try:
            sae_release = self.config.get('sae_release', 'gemma-scope-2b-pt-res-canonical')
            sae_width = self.config.get('sae_width', '16k')
            sae_type = self.config.get('sae_type', 'canonical')
            sae_id = f"layer_{self.layer}/width_{sae_width}/{sae_type}"
            
            print(f"Loading SAE: {sae_release} / {sae_id}")
            
            self.sae, self.sae_cfg_dict, _ = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=self.device
            )
            self.sae = self.sae.to(self.device)
            print("SAE loaded successfully")
        except Exception as e:
            print(f"Error loading SAE: {e}")
            raise
    
    def create_ablation_hook(self, feature_idx: int, position: int):
        """Create hook for ablating a specific feature at a position"""
        def hook_fn(acts: torch.Tensor, hook):
            if hook.name != self.hook_point:
                print(f"{hook.name} != {self.hook_point}!!")
                return acts
            
            acts_clone = acts.clone()

            # Encode position activation to SAE features
            pos_act = acts[0, position, :].unsqueeze(0)
            sae_features = self.sae.encode(pos_act)
            
            # Zero out the target feature
            sae_features_ablated = sae_features.clone()
            sae_features_ablated[0, feature_idx] = 0.0
            
            # Decode back to activation space
            reconstructed = self.sae.decode(sae_features_ablated)
            acts_clone[0, position, :] = reconstructed.squeeze(0)
            
            return acts_clone
        
        return (self.hook_point, hook_fn)
    
    def get_logits_range(self, tokens: torch.Tensor, start_pos: int, 
                        num_positions: int, hooks=None) -> torch.Tensor:
        """Get model logits for a range of positions"""
        with torch.no_grad():
            if hooks:
                output = self.model.run_with_hooks(tokens, fwd_hooks=hooks)
            else:
                output = self.model(tokens)
        
        logits = output[0] if isinstance(output, tuple) else output
        logits = logits[0]  # Get batch 0

        return logits[start_pos:start_pos + num_positions]
    
    def tokenize_sequence(self, text: str) -> List[int]:
        """Tokenize a text sequence"""
        tokens = self.model.to_tokens(text).squeeze(0)
        return tokens.tolist()
    
    def analyze_sequence(self, feature_idx: int, token_ids: List[int], ablation_pos: int) -> Dict:
        """Analyze effect of feature on a single sequence"""
        
        # Prepare tokens tensor
        tokens_tensor = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        max_depth = self.config["max_depth"]
        
        # Get base predictions
        base_logits = self.get_logits_range(tokens_tensor, ablation_pos, max_depth)
        
        ablation_hook = self.create_ablation_hook(feature_idx, ablation_pos)

        ablated_logits = self.get_logits_range(
            tokens_tensor, ablation_pos, max_depth, hooks=[ablation_hook]
        )
        
        # Extract predictions
        base_preds = []
        ablated_preds = []
        
        result = []

        for i in range(max_depth):
            base_token = self.model.tokenizer.decode([torch.argmax(base_logits[i]).item()])
            ablated_token = self.model.tokenizer.decode([torch.argmax(ablated_logits[i]).item()])
            
            base_preds.append(base_token)
            ablated_preds.append(ablated_token)
            
            # Calculate detailed effects
            base_probs = F.softmax(base_logits[i].float(), dim=-1)
            ablated_probs = F.softmax(ablated_logits[i].float(), dim=-1)
            
            # KL divergence
            log_ablated = F.log_softmax(ablated_logits[i].float(), dim=-1)
            kl_div = F.kl_div(log_ablated, base_probs, reduction='sum').item()
            
            # Top probability changes
            prob_diff = ablated_probs - base_probs
            top_inc_vals, top_inc_idx = torch.topk(prob_diff, 3)
            top_dec_vals, top_dec_idx = torch.topk(-prob_diff, 3)
            
            result.append({
                "position": f"t+{i+1}",
                "base_pred": base_token,
                "ablated_pred": ablated_token,
                "changed": base_token != ablated_token,
                "kl_divergence": kl_div if np.isfinite(kl_div) else float('inf'),
                "top_increases": [
                    (self.model.tokenizer.decode([idx.item()]), val.item())
                    for idx, val in zip(top_inc_idx, top_inc_vals)
                ],
                "top_decreases": [
                    (self.model.tokenizer.decode([idx.item()]), val.item())
                    for idx, val in zip(top_dec_idx, top_dec_vals)
                ]
            })

        return result
