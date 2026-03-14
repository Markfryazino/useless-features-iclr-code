import torch
import torch.nn as nn
from src.patched_gpt2 import GPT2Config, GPT2Model, GPT2LMHeadModel
from typing import Optional, Dict, Any, Tuple, List
import math


def myopic_forward(model, myopic_on=True, *args, **kwargs):
    initial_myopic_setting = GPT2LMHeadModel.MYOPIC
    assert model.config._attn_implementation == "eager"

    GPT2LMHeadModel.MYOPIC = myopic_on
    result = model(*args, **kwargs)
    GPT2LMHeadModel.MYOPIC = initial_myopic_setting
    return result


class HFDecoderModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_len: int,
        num_layers: int,
        vocab_size: int = 5,
        n_head: int = 2,
        feedforward_dim: int = None,
        myopic: bool = False,
        eager_attn: bool = False
    ):
        super().__init__()
        
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_positions=max_len,
            n_ctx=max_len,
            n_head=n_head,
            feedforward_dim=feedforward_dim or hidden_dim,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            summary_first_dropout=0
        )
        if myopic:
            GPT2LMHeadModel.MYOPIC = True
        if eager_attn:
            self.config._attn_implementation = "eager"

        self.transformer = GPT2LMHeadModel(self.config)

    def forward(
        self, 
        x: torch.Tensor, 
        intervene: Optional[Dict[str, Any]] = None,
        return_hidden: bool = False,
        return_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        detach_position: Optional[int] = None,
        detach_layer: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional intervention on hidden states (residual).
        
        Args:
            x: input tensor of shape (batch_size, seq_len)
            intervene: dictionary with possible keys:
                - 'layer_idx' (int): which layer index to modify (0-based)
                - 'intervene' (bool): whether we do an intervention
                - 'values' (torch.Tensor): the tensor to add into the hidden states
            return_hidden (bool): whether to return hidden states
            
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: 
                logits, optionally stacked hidden states
        """
        do_intervene = False
        target_layer = None
        intervention_values = None
        
        if intervene is not None:
            do_intervene = intervene.get('intervene', False)
            target_layer = intervene.get('layer_idx', None)
            intervention_values = intervene.get('values', None)

        if target_layer == "all":
            result = self.transformer(
                x, 
                return_dict=True, 
                output_hidden_states=True,
                block_intervention_layer="all",
                intervention_values=intervention_values,
                use_cache=not do_intervene,
                past_key_values=past_key_values,
                detach_position=detach_position,
                detach_layer=detach_layer
            )

        else:
            if do_intervene and target_layer == 0:
                raise ValueError("Intervening on layer_idx=0 (embedding) is not supported in this model.")
            
            block_index = None
            if do_intervene and (target_layer is not None):
                block_index = target_layer - 1
                # Make sure it's within [0..num_layers-1]
                if block_index < 0 or block_index >= self.config.n_layer:
                    raise ValueError(f"Requested layer_idx={target_layer} is out of range for {self.config.n_layer} blocks.")

            # Run the forward pass with HF
            result = self.transformer(
                x, 
                return_dict=True, 
                output_hidden_states=True,
                block_intervention_layer=block_index,
                intervention_values=intervention_values,
                use_cache=not do_intervene,
                past_key_values=past_key_values,
                detach_position=detach_position,
                detach_layer=detach_layer
            )

        logits = result.logits

        if return_cache:
            return logits, result.past_key_values
        elif return_hidden:
            # result.hidden_states is a tuple of hidden states at each layer plus final
            # shape: (num_layers+1, batch_size, seq_len, hidden_dim)
            return logits, torch.stack(result.hidden_states, dim=0)
        else:
            return logits


# split_index -- index of the first token to go into the second transformer
class HFSplitDecoderModel(HFDecoderModel):
    def __init__(self, split_index: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_index = split_index
        self.transformer2 = GPT2LMHeadModel(self.config)

    def forward(
        self, 
        x: torch.Tensor, 
        intervene: Optional[Dict[str, Any]] = None,
        return_hidden: bool = False,
        return_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        detach_position: Optional[int] = None,
        detach_layer: Optional[int] = None
    ):
        if intervene is not None:
            raise NotImplementedError("Intervention not supported for HFSplitDecoderModel")

        if x.size(1) <= self.split_index:
            return super().forward(x, intervene, return_hidden)
        else:
            x1 = x[:, :self.split_index]
            x2 = x[:, self.split_index:]

            result1 = self.transformer(x1, return_dict=True, output_hidden_states=True, use_cache=True, past_key_values=past_key_values, detach_position=detach_position, detach_layer=detach_layer)
            result2 = self.transformer2(x2, past_key_values=result1.past_key_values, return_dict=True, output_hidden_states=True, use_cache=True, detach_position=detach_position, detach_layer=detach_layer)
            if return_cache:
                return torch.cat([result1.logits, result2.logits], dim=1), \
                       result2.past_key_values
            elif return_hidden:
                hidden1 = torch.stack(result1.hidden_states, dim=0)
                hidden2 = torch.stack(result2.hidden_states, dim=0)
                return torch.cat([result1.logits, result2.logits], dim=1), \
                       torch.cat([hidden1, hidden2], dim=2)
            else:
                return torch.cat([result1.logits, result2.logits], dim=1)


def get_hf_model(*args, **kwargs):
    if kwargs.get('split_index', None) == "all":
        del kwargs["split_index"]
        return HFUntiedDecoderModel(*args, **kwargs)
    elif (kwargs.get('split_index', None) is not None) and isinstance(kwargs['split_index'], (int, float, str)) and str(kwargs['split_index']).isdigit():
        kwargs['split_index'] = int(kwargs['split_index'])
        return HFSplitDecoderModel(*args, **kwargs)
    else:
        if "split_index" in kwargs:
            del kwargs["split_index"]
        return HFDecoderModel(*args, **kwargs)
    

class HFUntiedDecoderModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        max_len: int,
        num_layers: int,
        vocab_size: int = 5,
        n_head: int = 2,
        feedforward_dim: int = None,
        myopic: bool = False,
        eager_attn: bool = False
    ):
        super().__init__()
        
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_positions=max_len,
            n_ctx=max_len,
            n_head=n_head,
            feedforward_dim=feedforward_dim or hidden_dim,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            summary_first_dropout=0
        )
        if myopic:
            GPT2LMHeadModel.MYOPIC = True
        if eager_attn:
            self.config._attn_implementation = "eager"

        print(f"Myopic: {GPT2LMHeadModel.MYOPIC}, Eager attn: {eager_attn}")

        # Initialize the first transformer
        first_transformer = GPT2LMHeadModel(self.config)
        
        # Create the list of transformers, starting with the first one
        self.transformers = nn.ModuleList([first_transformer])
        
        # Add the rest of the transformers, initializing them with the state_dict of the first one
        for _ in range(1, max_len):
            new_transformer = GPT2LMHeadModel(self.config)
            new_transformer.load_state_dict(first_transformer.state_dict())
            self.transformers.append(new_transformer)

    def forward(
        self, 
        x: torch.Tensor, 
        intervene: Optional[Dict[str, Any]] = None,
        return_hidden: bool = False,
        return_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        detach_position: Optional[int] = None,
        detach_layer: Optional[int] = None
    ) -> torch.Tensor:
        if intervene is not None:
            raise NotImplementedError("Intervention not supported for HFUntiedDecoderModel")

        past_kv = past_key_values
        hidden_states = None
        logits = None
        i = 0 if past_kv is None else past_kv[0][0].shape[2]
        for j in range(i, i + x.shape[1]):
            result = self.transformers[j](
                x[:, j - i : j - i+1], 
                past_key_values=past_kv, 
                return_dict=True, 
                output_hidden_states=True, 
                use_cache=True,
                detach_position=detach_position,
                detach_layer=detach_layer
            )
            past_kv = result.past_key_values

            if logits is not None:
                logits = torch.cat([logits, result.logits], dim=1)
                hidden_states = torch.cat(
                    [hidden_states, torch.stack(result.hidden_states, dim=0)], dim=2
                )
            else:
                logits = result.logits
                hidden_states = torch.stack(result.hidden_states, dim=0)

        if return_cache:
            return logits, past_kv
        elif return_hidden:
            return logits, hidden_states
        else:
            return logits
