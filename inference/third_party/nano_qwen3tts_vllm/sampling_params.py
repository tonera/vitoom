from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
