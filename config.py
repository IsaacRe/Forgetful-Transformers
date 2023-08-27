from dataclasses import dataclass, field
import argparse
from typing import List

@dataclass
class Config:
    batch_size: int = 8
    effective_batch_size: int = 64
    lr: float = 1e-3
    lrd_gamma: float = 0.1
    lrd_steps: List[int] = field(default_factory=[5000, 14000])
    eval_interval: int = 50
    save_interval: int = 50
    wandb_api_key: str = None
    experiment_id: str = None
    do_consolidate: bool = True
    context_length: int = 400
    consolidate_length: int = 200
    train_length: int = 200
    consolidate_ratio: float = 0.5
    device: str = "cuda:0"
    scale_v: bool = False

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--effective-batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lrd-gamma", default=0.1, type=float)
parser.add_argument("--lrd-steps", nargs="*", default=[5000, 14000], type=int)
parser.add_argument("--eval-interval", default=50, type=int)
parser.add_argument("--save-interval", default=50, type=int)
parser.add_argument("--wandb-api-key", default=None, type=str)
parser.add_argument("--experiment-id", default="QKNorm")
parser.add_argument("--scale-v", action="store_true")
args, _ = parser.parse_known_args()

CONFIG = Config(
    batch_size=args.batch_size,
    effective_batch_size=args.effective_batch_size,
    lr=args.lr,
    lrd_gamma=args.lrd_gamma,
    lrd_steps=args.lrd_steps,
    eval_interval=args.eval_interval,
    save_interval=args.save_interval,
    wandb_api_key=args.wandb_api_key,
    experiment_id=args.experiment_id,
    scale_v=args.scale_v
)
