from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from config import SHARED_DIR, TokenConfig, TokenizationConfig


def iter_stories(
    tokenization_config: TokenizationConfig, path: Path, limit: int | None = None
) -> Iterable[str]:
    """Yields stories from the txt file one at a time"""
    buffer = ""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            pieces = buffer.split(tokenization_config.story_delimiter)
            buffer = pieces.pop()
            for piece in pieces:
                story = piece.strip()
                if not story:
                    continue
                yield story
                count += 1
                if limit is not None and count >= limit:
                    return
        tail = buffer.strip()
        if tail and (limit is None or count < limit):
            yield tail


def count_tokens(
    tokenization_config: TokenizationConfig,
    tokenizer: Tokenizer,
    path: Path,
    story_limit: int | None = None,
) -> int:
    total = 0
    for story in iter_stories(tokenization_config, path, limit=story_limit):
        total += len(tokenizer.encode(story).ids) + 1  # +1 for the EOS Token!
    return total


def build_token_memmap(
    token_config: TokenConfig,
    tokenizer: Tokenizer,
    path: Path,
    total_tokens: int,
    output_path: Path,
    story_limit: int | None = None,
) -> Path:
    """Returns the path of the built token memmap

    A memmap is basically like a token stream, but optimized for quick retrieval"""
    if output_path.exists():
        return output_path

    # grab the EOS ID
    eos_id = tokenizer.token_to_id(token_config.eos)
    assert eos_id is not None

    # build the memmap, defining the shape up front
    token_array = np.memmap(
        output_path, dtype=np.uint32, mode="w+", shape=(total_tokens,)
    )

    # iterate through the dataset and build it as a stream of tokens
    offset = 0
    for story in iter_stories(path, limit=story_limit):
        ids = tokenizer.encode(story).ids

        # force a gap here
        next_offset = offset + len(ids) + 1
        token_array[offset : offset + len(ids)] = ids

        # put the EOS token in the gap
        token_array[offset + len(ids)] = eos_id
        offset = next_offset
    token_array.flush()

    return output_path


def build_tokenizer(
    tokenization_config: TokenizationConfig,
    token_config: TokenConfig,
    train_path: Path,
    vocab_size: int,
    story_limit: int,
) -> Tokenizer:
    """Trains a tokenizer based on the inputted data"""
    # initialize
    tokenizer = Tokenizer(models.BPE(unk_token=token_config.unk))

    # define the metaspace and decoders
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁", prepend_scheme="always"
    )
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

    # init the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=[
            token_config.bos,
            token_config.eos,
            token_config.pad,
            token_config.unk,
        ],
    )

    # train!
    tokenizer.train_from_iterator(
        iter_stories(tokenization_config, train_path, limit=story_limit),
        trainer=trainer,
        length=story_limit,
    )
    tokenizer.save(
        str(SHARED_DIR / f"tinystories_bpe_metaspace_{vocab_size}_{story_limit}.json")
    )
    return tokenizer
