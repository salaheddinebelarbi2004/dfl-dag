"""
Model Registry

Provides a small registry to register and instantiate model loaders/wrappers
so the system can be model-agnostic. Conventions:

- Register a loader callable that accepts keyword args and returns a model
  instance (or a loaded object). Example loader signature:
    def loader(model_path: str, **kwargs) -> Any: ...

- The registry does not enforce framework types; callers decide how to
  interpret the returned object.

Usage:
    from model_registry import registry
    model = registry.get('pytorch', model_path='model.pt', wrapper_class=MyWrapper)

This file ships with a basic `pytorch` loader that can either return the
raw object from `torch.load()` or instantiate a provided wrapper class.
"""
from typing import Any, Callable, Dict, Optional


class ModelRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, constructor: Callable[..., Any]) -> None:
        """Register a constructor/loader under `name`.

        The constructor should be a callable that returns a model/object when
        called with appropriate kwargs.
        """
        key = name.lower()
        self._registry[key] = constructor

    def get(self, name: str, **kwargs) -> Any:
        """Instantiate/load the model registered under `name`.

        Raises KeyError if no loader is registered for `name`.
        """
        key = name.lower()
        if key not in self._registry:
            raise KeyError(f"No model loader registered under '{name}'")
        return self._registry[key](**kwargs)

    def has(self, name: str) -> bool:
        return name.lower() in self._registry

    def list_models(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._registry)

    def unregister(self, name: str) -> None:
        key = name.lower()
        if key in self._registry:
            del self._registry[key]


registry = ModelRegistry()


def pytorch_loader(model_path: str, wrapper_class: Optional[Callable[..., Any]] = None, map_location: Optional[str] = None, **kwargs) -> Any:
    """Basic PyTorch loader helper.

    - If `wrapper_class` is provided, attempts to instantiate it using
      `wrapper_class(model_path=model_path, **kwargs)`. If that fails due to
      signature mismatch, it will load the state dict with `torch.load()` and
      pass `state_dict=...` to the wrapper.
    - If no `wrapper_class` is provided, returns the raw result of
      `torch.load(model_path, map_location=map_location)`.

    Note: importing torch is done inside the function to avoid requiring torch
    when the registry is imported for other lightweight use-cases.
    """
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for the 'pytorch' loader") from e

    if wrapper_class is not None:
        # Prefer wrapper that accepts model_path directly
        try:
            return wrapper_class(model_path=model_path, **kwargs)
        except TypeError:
            # Fallback: load state dict and attempt to call wrapper with it
            state = torch.load(model_path, map_location=map_location)
            return wrapper_class(state_dict=state, **kwargs)
    # No wrapper class: return raw torch.load output
    return torch.load(model_path, map_location=map_location)


# Register default loaders
registry.register('pytorch', pytorch_loader)


def safetensors_loader(model_path: str, wrapper_class: Optional[Callable[..., Any]] = None, map_location: Optional[str] = None, **kwargs) -> Any:
    """Loader for .safetensors files using safetensors (torch bridge).

    Attempts to use `safetensors.torch.load_file`. Returns the raw dict of
    tensors unless a `wrapper_class` is provided, in which case it will try
    to instantiate the wrapper with `model_path` or `state_dict=` as fallback.
    """
    try:
        # Import lazily to avoid hard dependency unless used
        import safetensors.torch as st
    except Exception as e:
        raise RuntimeError("safetensors is required for the 'safetensors' loader") from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required for using safetensors loader") from e

    # If caller supplied a wrapper that can accept model_path, prefer it
    if wrapper_class is not None:
        # For safetensors, prefer to construct the wrapper from an in-memory
        # state_dict to avoid invoking framework-specific file loaders inside
        # the wrapper (which may call torch.load and fail for safetensors).
        state = st.load_file(model_path)
        state_torch = {k: torch.as_tensor(v) for k, v in state.items()}
        return wrapper_class(state_dict=state_torch, **kwargs)

    # No wrapper: return raw safetensors-loaded dict (torch tensors)
    state = st.load_file(model_path)
    state_torch = {k: torch.as_tensor(v) for k, v in state.items()}
    return state_torch


def numpy_loader(model_path: str, wrapper_class: Optional[Callable[..., Any]] = None, **kwargs) -> Any:
    """Loader for .npy files.

    Returns numpy array. If wrapper_class provided, passes array to it.
    """
    import numpy as np

    model = np.load(model_path)

    if wrapper_class is not None:
        return wrapper_class(model=model, **kwargs)

    return model


def numpy_compressed_loader(model_path: str, wrapper_class: Optional[Callable[..., Any]] = None, **kwargs) -> Any:
    """Loader for .npz files.

    Attempts to extract 'model' key or returns first array. If wrapper_class
    provided, passes extracted array to it.
    """
    import numpy as np

    data = np.load(model_path)

    # Try to extract from 'model' key or use first array
    if isinstance(data, np.ndarray):
        model = data
    elif 'model' in data:
        model = data['model']
    else:
        model = data[list(data.keys())[0]]

    if wrapper_class is not None:
        return wrapper_class(model=model, **kwargs)

    return model


# Register all loaders
registry.register('pytorch', pytorch_loader)
registry.register('safetensors', safetensors_loader)
registry.register('numpy', numpy_loader)
registry.register('numpy_compressed', numpy_compressed_loader)


__all__ = ["ModelRegistry", "registry", "pytorch_loader", "safetensors_loader", "numpy_loader", "numpy_compressed_loader"]
