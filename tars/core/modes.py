# tars/core/modes.py

from enum import Enum
import random


class ReasoningMode(str, Enum):
    ANALYST = "evidence-centered analyst"
    CRITIC = "devil's-advocate critic"
    SYNTHESIZER = "synthetic innovator"


def random_mode() -> ReasoningMode:
    return random.choice(list(ReasoningMode))
