from dataclasses import dataclass
from pathlib import Path

# DIRECTORY STUFF
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

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
TOKENIZER_DIR = ARTIFACTS_DIR / "tokenizers"


@dataclass
class DirectoryConfig:
    data: str = DATA_DIR
    artifacts: str = ARTIFACTS_DIR
    models: str = MODELS_DIR
    plots: str = PLOTS_DIR
    tokenizers: str = TOKENIZER_DIR


DIRECTORIES = DirectoryConfig()

# DATA STUFF
TRAIN_FILENAME = "TinyStoriesV2-GPT4-train.txt"
VALID_FILENAME = "TinyStoriesV2-GPT4-valid.txt"
STORY_DELIMITER = "<|endoftext|>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


@dataclass
class DataConfig:
    training_file: str = TRAIN_FILENAME
    validation_file: str = VALID_FILENAME
    story_delimiter: str = STORY_DELIMITER
    special_tokens: tuple[str] = SPECIAL_TOKENS


DATA = DataConfig()

# TRAINING STUFF
SEED = 242
MAX_TRAIN_STORIES = 200_000
VOCAB_SIZE = 3_000
CONTEXT_LENGTH = 128
CHECKPOINT_EVERY = 100
SAMPLE_PROMPTS = (
    "This tale begins with an ogre in a swamp. ",
    "The boring suburban life of Robert Parr, once Mr. Incredible",
    "Manny the mammoth befriended a sloth and a sabre toothed tiger",
    "Woody and Buzz Lightyear, two of Andy's toys had come alive!",
    "There was no car faster than Lightning McQueen",
)


@dataclass
class TrainingConfig:
    seed: int = 242
    max_train_stories: int = MAX_TRAIN_STORIES
    vocab_size: int = VOCAB_SIZE
    context_window: int = CONTEXT_LENGTH
    checkpoint_every: int = CHECKPOINT_EVERY
    sample_prompts: tuple[str] = SAMPLE_PROMPTS


TRAINING_CONFIG = TrainingConfig()
