from model_registry import registry
from run_with_shards import RealModelWrapper
import os, traceback
print('loaders=', list(registry.list_models().keys()))
print('model.safetensors exists=', os.path.exists('model.safetensors'))
try:
    # Load raw safetensors to inspect keys
    import safetensors.torch as st
    s = st.load_file('model.safetensors')
    print('safetensors keys:', list(s.keys()))
    for k,v in s.items():
        print(k, type(v), getattr(v, 'shape', None))

    m = registry.get('safetensors', model_path='model.safetensors', wrapper_class=RealModelWrapper, learning_rate=0.001)
    print('instantiated:', type(m))
except Exception as e:
    traceback.print_exc()
