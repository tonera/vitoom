"""Utility functions for weight loading and transformation."""
import torch
from typing import Dict


def transform_predictor_weights(
    state_dict: Dict[str, torch.Tensor],
    config,
) -> Dict[str, torch.Tensor]:
    """Transform predictor weights from original format to model format.
    
    Transformations:
    1. Merge q_proj, k_proj, v_proj -> qkv_proj (order: Q, K, V)
    2. Merge gate_proj, up_proj -> gate_up_proj (order: gate, up)
    3. Direct mapping for all other weights
    
    Args:
        state_dict: Original state dictionary
        config: Model configuration
        
    Returns:
        Transformed state dictionary
    """
    transformed_dict = {}
    processed_keys = set()
    
    # Find all layer indices
    layer_indices = set()
    for key in state_dict.keys():
        if ".layers." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_indices.add(layer_idx)
                        break
                    except ValueError:
                        pass
    
    # Process each layer
    for layer_idx in sorted(layer_indices):
        layer_prefix = f"model.layers.{layer_idx}"
        
        # Process attention weights: q_proj, k_proj, v_proj -> qkv_proj
        q_proj_key = f"{layer_prefix}.self_attn.q_proj.weight"
        k_proj_key = f"{layer_prefix}.self_attn.k_proj.weight"
        v_proj_key = f"{layer_prefix}.self_attn.v_proj.weight"
        
        if q_proj_key in state_dict and k_proj_key in state_dict and v_proj_key in state_dict:
            # Merge QKV: concatenate along output dimension (dim=0)
            qkv_weight = torch.cat([
                state_dict[q_proj_key],
                state_dict[k_proj_key],
                state_dict[v_proj_key]
            ], dim=0)
            transformed_dict[f"{layer_prefix}.self_attn.qkv_proj.weight"] = qkv_weight
            processed_keys.update([q_proj_key, k_proj_key, v_proj_key])
        
        # Process MLP weights: gate_proj, up_proj -> gate_up_proj
        gate_proj_key = f"{layer_prefix}.mlp.gate_proj.weight"
        up_proj_key = f"{layer_prefix}.mlp.up_proj.weight"
        
        if gate_proj_key in state_dict and up_proj_key in state_dict:
            # Merge gate and up: concatenate along output dimension (dim=0)
            gate_up_weight = torch.cat([
                state_dict[gate_proj_key],
                state_dict[up_proj_key]
            ], dim=0)
            transformed_dict[f"{layer_prefix}.mlp.gate_up_proj.weight"] = gate_up_weight
            processed_keys.update([gate_proj_key, up_proj_key])
    
    # Copy all other keys directly
    for key in state_dict.keys():
        if key not in processed_keys:
            transformed_dict[key] = state_dict[key]
    
    return transformed_dict


def transform_talker_weights(
    state_dict: Dict[str, torch.Tensor],
    config,
) -> Dict[str, torch.Tensor]:
    """Transform talker weights from original format to model format.
    
    Transformations:
    1. Remove 'talker.' prefix from all keys
    2. Merge q_proj, k_proj, v_proj -> qkv_proj (order: Q, K, V)
    3. Merge gate_proj, up_proj -> gate_up_proj (order: gate, up)
    4. Include text_embedding and text_projection weights
    5. Direct mapping for all other weights
    
    Args:
        state_dict: Original state dictionary with 'talker.' prefix
        config: Model configuration
        
    Returns:
        Transformed state dictionary
    """
    # First, remove 'talker.' prefix
    prefixed_dict = {}
    for key, value in state_dict.items():
        if key.startswith("talker."):
            new_key = key.replace("talker.", "", 1)
            prefixed_dict[new_key] = value
        else:
            prefixed_dict[key] = value
    
    transformed_dict = {}
    processed_keys = set()
    
    # Find all layer indices
    layer_indices = set()
    for key in prefixed_dict.keys():
        if ".layers." in key:
            parts = key.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        layer_indices.add(layer_idx)
                        break
                    except ValueError:
                        pass
    
    # Process each layer
    for layer_idx in sorted(layer_indices):
        layer_prefix = f"model.layers.{layer_idx}"
        
        # Process attention weights: q_proj, k_proj, v_proj -> qkv_proj
        q_proj_key = f"{layer_prefix}.self_attn.q_proj.weight"
        k_proj_key = f"{layer_prefix}.self_attn.k_proj.weight"
        v_proj_key = f"{layer_prefix}.self_attn.v_proj.weight"
        
        if q_proj_key in prefixed_dict and k_proj_key in prefixed_dict and v_proj_key in prefixed_dict:
            # Merge QKV: concatenate along output dimension (dim=0)
            qkv_weight = torch.cat([
                prefixed_dict[q_proj_key],
                prefixed_dict[k_proj_key],
                prefixed_dict[v_proj_key]
            ], dim=0)
            transformed_dict[f"{layer_prefix}.self_attn.qkv_proj.weight"] = qkv_weight
            processed_keys.update([q_proj_key, k_proj_key, v_proj_key])
        
        # Process MLP weights: gate_proj, up_proj -> gate_up_proj
        gate_proj_key = f"{layer_prefix}.mlp.gate_proj.weight"
        up_proj_key = f"{layer_prefix}.mlp.up_proj.weight"
        
        if gate_proj_key in prefixed_dict and up_proj_key in prefixed_dict:
            # Merge gate and up: concatenate along output dimension (dim=0)
            gate_up_weight = torch.cat([
                prefixed_dict[gate_proj_key],
                prefixed_dict[up_proj_key]
            ], dim=0)
            transformed_dict[f"{layer_prefix}.mlp.gate_up_proj.weight"] = gate_up_weight
            processed_keys.update([gate_proj_key, up_proj_key])
    
    # Copy all other keys directly (including text_embedding and text_projection)
    for key in prefixed_dict.keys():
        if key not in processed_keys:
            transformed_dict[key] = prefixed_dict[key]
    
    return transformed_dict
