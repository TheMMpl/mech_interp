from typing import List, Optional
import torch

class InjectionHook:
    def __init__(self, model, layer_idx, steering_vectors, injection_position: Optional[int] = None):
        self.model = model
        self.layer_idx = layer_idx
        self.vectors = steering_vectors
        self.injection_position = injection_position
        self.handle = None

    def _hook(self, module, inputs, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output

        # Aggregate vectors
        total_delta = torch.zeros_like(self.vectors[0][0]).to(h.device)
        for vec, strength in self.vectors:
            total_delta += strength * vec.to(h.device)

        # Inject
        if self.injection_position is not None:
            # Inject at specific token position
            if self.injection_position < h.shape[1]:
                #print(h[:, self.injection_position, :].shape, total_delta.shape, self.injection_position)
                h[:, self.injection_position, :] += total_delta
        else:
            # Broadcast inject (all positions)
            h = h + total_delta.view(1, 1, -1)

        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    def _get_layers(self, model):
        if hasattr(model, "model") and hasattr(model.model, "layers"): return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"): return model.transformer.h
        if hasattr(model, "layers"): return model.layers
        if hasattr(model, "base_model"): return self._get_layers(model.base_model)
        raise AttributeError(f"Cannot find layers in model of type {type(model)}")

    def __enter__(self):
        layers = self._get_layers(self.model)
        self.handle = layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle: self.handle.remove()

