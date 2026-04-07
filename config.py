import os
from dataclasses import dataclass
from pathlib import Path

if "__file__" in globals():
    ROOT = Path(__file__).resolve().parent
else:
    _cwd = Path.cwd().resolve()
    if _cwd.name == "deliverable" and (_cwd.parent / "data").exists():
        ROOT = _cwd.parent
    elif (_cwd / "hw3").exists():
        ROOT = _cwd / "hw3"
    else:
        ROOT = _cwd

# Directory level configuration
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
SHARED_DIR = ARTIFACTS_DIR / "shared"
RUN_DIR = ARTIFACTS_DIR / "runs"
COLAB_DIR = Path("/content/drive/MyDrive/courses/242B/HW3")

# Data files and configuration
TRAIN_FILENAME = "TinyStoriesV2-GPT4-train.txt"
VALID_FILENAME = "TinyStoriesV2-GPT4-valid.txt"


class RunConfig:
    """A configuration object to handle all the run-specific information"""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

        # create the directory if it doesn't exist
        if os.path.isdir(RUN_DIR / run_id):
            raise FileExistsError(f"Directory already exists: {RUN_DIR / run_id}")
        else:
            os.makedirs(RUN_DIR / run_id)
            os.makedirs(RUN_DIR / run_id / "metrics")
            os.makedirs(RUN_DIR / run_id / "models")
            os.makedirs(RUN_DIR / run_id / "plots")
            print(f"Created {RUN_DIR / run_id} and subfolders")

        self.metrics = RUN_DIR / run_id / "metrics"
        self.models = RUN_DIR / run_id / "models"
        self.plots = RUN_DIR / run_id / "plots"


@dataclass
class DataConfig:
    training_file_colab: str = COLAB_DIR / TRAIN_FILENAME
    training_file_local: str = COLAB_DIR / TRAIN_FILENAME
    validation_file_colab: str = DATA_DIR / VALID_FILENAME
    validation_file_local: str = DATA_DIR / VALID_FILENAME


@dataclass
class TokenConfig:
    eos: str = "<eos>"
    pad: str = "<pad>"
    bos: str = "<bos>"
    unk: str = "<unk>"


@dataclass
class GlobalTrainingConfig:
    context_length: int = 128
    checkpoint_every: int = 100


@dataclass
class TokenizationConfig:
    story_delimiter: str = "<|endoftext|>"
    vocab_size: int = 3_000
    max_train_stories: int = 1_000_000


@dataclass
class ModelConfig:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    dropout: float = 0.1
    grad_clip_norm: float = 1.0
    use_amp: bool = True
