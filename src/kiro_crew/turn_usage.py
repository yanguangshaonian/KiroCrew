"""各模型运行时共享的单轮用量记录."""

from dataclasses import dataclass


@dataclass
class TurnUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    credits: float = 0.0
    num_turns: int = 0
    duration_ms: int = 0
