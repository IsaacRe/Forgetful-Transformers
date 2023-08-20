from dataclasses import dataclass

@dataclass
class Config:
    do_consolidate: bool = True
    context_length: int = 400
    consolidate_length: int = 200
    train_length: int = 200
    consolidate_ratio: float = 0.5
    device: str = "cuda:0"


CONFIG = Config()
