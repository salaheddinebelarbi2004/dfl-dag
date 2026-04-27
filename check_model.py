"""
Federated Learning with REAL Model.pt and Real Embeddings
Each client gets different data slice
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Check model
model_state = torch.load('model.pt')
print('Model loaded!')
print(f'Type: {type(model_state)}')
print(f'Keys: {list(model_state.keys())}')

for key, val in model_state.items():
    if hasattr(val, 'shape'):
        print(f'  {key}: {val.shape}')
    else:
        print(f'  {key}: {type(val)}')
