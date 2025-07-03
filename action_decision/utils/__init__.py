from typing import NamedTuple
import torch


class Transition(NamedTuple):
    observation: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    done: bool
    hole_pos: str
