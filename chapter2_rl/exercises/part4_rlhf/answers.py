# %%
import torch as t
import torch.nn as nn
from torch import Tensor
import wandb
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils, HookedTransformer
from typing import List, Optional, Tuple, Union, Dict, Any, Callable
import einops
from jaxtyping import Float, Int
import os
import sys
from pathlib import Path
from rich import print as rprint
from rich.table import Table
from eindex import eindex
from dataclasses import dataclass
import numpy as np
import time
from functools import partial

# Make sure exercises are in the path
chapter = r"chapter2_rl"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_rlhf"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

import part4_rlhf.tests as tests
import part4_rlhf.solutions as solutions

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

LOW_GPU_MEM = True
BASE_MODEL = "gpt2-small" if LOW_GPU_MEM else "gpt2-medium"
# %%
class TransformerWithValueHead(nn.Module):
    '''
    Defines a GPT model with a value head (the latter taking the last hidden state as input,
    post-layernorm).

    The value head is a simple MLP with one hidden layer, and scalar output:

        Linear(d_model -> 4*d_model)
        ReLU
        Linear(4*d_model -> 1)

    All linear layers have biases.
    '''
    base_model: HookedTransformer
    value_head: nn.Sequential

    def __init__(self, base_model: str = BASE_MODEL):
        super().__init__()

        self.base_model = HookedTransformer.from_pretrained(base_model)

        d_model = self.base_model.cfg.d_model

        self.value_head = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, 1)
        )
        self.value_head_output = None

    def forward(self, input_ids: Int[Tensor, "batch seq"]) -> Tuple[
        Float[Tensor, "batch seq d_vocab"],
        Int[Tensor, "batch seq"]
    ]:

        def calc_and_store_value_head_output(resid_post: Float[Tensor, "batch seq d_model"], hook: HookPoint):
            self.value_head_output = self.value_head(resid_post).squeeze(-1)

        logits = self.base_model.run_with_hooks(
            input_ids,
            return_type = "logits",
            fwd_hooks = [
                (utils.get_act_name("normalized"), calc_and_store_value_head_output)
            ]
        )
        assert self.value_head_output is not None

        return logits, self.value_head_output

# Define a reference model (we'll use this during RLHF)
model = TransformerWithValueHead().to(device)

# Test your value head's architecture
assert isinstance(model.base_model, HookedTransformer), "Your model should have a HookedTransformer as its `base_model` attribute."
assert isinstance(model.value_head, nn.Sequential), "Your model should have a `value_head` attribute that is a `nn.Sequential`."
d_model = model.base_model.cfg.d_model
assert len(model.value_head) == 3, "Your value head should be a `nn.Sequential` with 3 layers."
assert sum(p.numel() for p in model.value_head.parameters()) == (d_model+1)*4*d_model + (4*d_model+1), "Your value head should have the correct number of parameters."

# Test your class's forward pass
input_ids = t.randint(0, 1000, (1, 10)).to(device)
logits, values = model(input_ids)
assert logits.shape == (*input_ids.shape, model.base_model.cfg.d_vocab), "Your model's logits should have shape (batch, seq, d_vocab)."
assert values.shape == input_ids.shape, "Your model's value head should give you an output for every token in your input. Did you forget to squeeze the out_features=1 dim?"

print("All tests for `TransformerWithValueHead` passed!")
# %%
