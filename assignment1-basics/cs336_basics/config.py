from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataConfig:
    # Memmap token files (1D binary files)
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    # Numpy dtype used when creating the token files
    np_dtype: str = "uint16"

    context_length: int = 256

    # Device string used by get_batch()
    device: str = "cuda:0"

@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # If None, will default to 4 * d_model at model construction time
    d_ff: Optional[int] = None

    rope_theta: float = 10_000.0
    # If None, model will use context_length
    max_seq_len: Optional[int] = None

    rmsnorm_eps: float = 1e-5

    # torch dtype string used for model parameters
    torch_dtype: str = "float32"

@dataclass
class OptimizerConfig:
    lr_max: float = 3e-4
    lr_min: float = 3e-5

    warmup_iters: int = 200
    cosine_cycle_iters: int = 10_000
    
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1

    grad_clip: float = 1.0

@dataclass
class TrainingConfig:
    max_steps: int = 10_000
    batch_size: int = 64
    
    log_interval: int = 50
    eval_interval: int = 500
    eval_batches: int = 20

    ckpt_interval: int = 1000
    ckpt_path: str = "checkpoints/ckpt.pt"
    resume_from: Optional[str] = None

    seed: int = 0

@dataclass
class WandbConfig:
    enable: bool = False
    project: str = "cs336-a1"
    run_name: str = "train"

@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

def get_default_config() -> TrainConfig:
    """
    Return a default training configuration.
    """
    cfg = TrainConfig()

    # Keep model/data context_length consistent by default
    cfg.model.context_length = cfg.data.context_length

    return cfg