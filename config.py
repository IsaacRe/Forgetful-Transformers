from dataclasses import dataclass, field
import argparse
from typing import List, Tuple

@dataclass
class Config:
    batch_size: int = 8
    effective_batch_size: int = 64
    lr: float = 1e-3
    lrd_gamma: float = 0.1
    lrd_steps: List[int] = field(default_factory=lambda: [5000, 14000])
    eval_interval: int = 50
    save_interval: int = 50
    wandb_api_key: str = None
    experiment_id: str = None
    do_consolidate: bool = True
    consolidate_method: str = "scissorhands"
    control_layers: List[int] = field(default_factory=lambda: [])
    context_length: int = 400
    consolidate_length: int = 200
    train_length: int = 200
    consolidate_range: List[int] = field(default_factory=lambda: [0, 200])
    fit_range: List[int] = field(default_factory=lambda: [200, 400])
    consolidate_ratio: float = 0.5
    min_fit_offset: int = 0
    device: str = "cuda:0"
    scale_v: bool = False

    def do_modify(self) -> bool:
        return self.do_consolidate

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--effective-batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--lrd-gamma", default=0.1, type=float)
parser.add_argument("--lrd-steps", nargs="*", default=[5000, 14000], type=int)
parser.add_argument("--eval-interval", default=50, type=int)
parser.add_argument("--save-interval", default=50, type=int)
parser.add_argument("--wandb-api-key", default=None, type=str)
parser.add_argument("--experiment-id", default="scissorhands")
parser.add_argument("--scale-v", action="store_true")
parser.add_argument("--consolidate-method", default="scissorhands", choices=["scissorhands"])
parser.add_argument("--control-layers", nargs="*", type=int, help="indices of layers to skip cache consolidation")
parser.add_argument("--do-consolidate", action="store_true")
parser.add_argument("--consolidate-range", nargs=2, default=[0, 200], help="range of keys to consolidate")
parser.add_argument("--consolidate-ratio", default=0.5, help="compression rate for range of keys being consolidated")
parser.add_argument("--fit-range", nargs=2, default=[200, 400], help="range of queries used to fit consolidation")
parser.add_argument("--min-fit-offset", default=0, help="minimum distance from consolidated keys to first query used to determine consolidation")

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
    do_consolidate=args.do_consolidate,
    consolidate_method=args.consolidate_method,
    control_layers=args.control_layers,
    consolidate_ratio=args.consolidate_ratio,
    consolidate_range=args.consolidate_range,
    fit_range=args.fit_range,
    min_fit_offset=args.min_fit_offset,
)
