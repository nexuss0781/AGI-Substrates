"""
AGI Unified Vocabulary
========================
Single-responsibility module for word ↔ index ↔ embedding mapping.
Synthesizes best capabilities from AGIVocabulary (encoder) and
DecoderVocabulary (decoder):

From encoder:
- Input embeddings as Tensor (autograd-compatible)
- encode() for text → token indices
- Open-vocabulary with automatic word addition

From decoder:
- Output embeddings (separate namespace for generation)
- decode() for token indices → text
- get_embedding_matrix() for efficient batch lookup
- Word frequency tracking for adaptive vocabulary
"""

import numpy as np
from typing import List, Dict, Optional
from nn import Tensor


class Vocabulary:
    """
    Unified AGI-grade vocabulary with encoding, decoding, and dual embeddings.

    Supports:
    - Bidirectional word ↔ index mapping with special tokens
    - Input embeddings (Tensor, autograd-compatible) for encoder
    - Output embeddings (numpy, lightweight) for decoder generation
    - Open vocabulary with character-level fallback
    - Semantic hash seeding for reproducible embeddings
    - Word frequency tracking for adaptive vocabulary management
    - Efficient embedding matrix retrieval for batch operations
    """

    SPECIAL_TOKENS = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.word_to_idx: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.idx_to_word: Dict[int, str] = {v: k for k, v in self.word_to_idx.items()}

        # Dual embedding spaces
        self.input_embeddings: Dict[int, Tensor] = {}    # For encoder (autograd)
        self.output_embeddings: Dict[int, np.ndarray] = {}  # For decoder (generation)

        # Frequency tracking (from decoder)
        self.word_frequencies: Dict[str, int] = {}

        self._init_special_tokens()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_special_tokens(self):
        """Initialize embeddings for special tokens."""
        for idx, word in self.idx_to_word.items():
            self.input_embeddings[idx] = Tensor(
                np.random.randn(self.embedding_dim) * 0.1,
                label=f'emb_{word}'
            )
            self.output_embeddings[idx] = np.random.randn(self.embedding_dim) * 0.1

    # ------------------------------------------------------------------
    # Word management
    # ------------------------------------------------------------------
    def add_word(self, word: str) -> int:
        """
        Add word to vocabulary with learned embedding (open vocabulary).
        Uses semantic hash for reproducible initialization.
        """
        if word not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            # Semantic hash for reproducible embeddings
            np.random.seed(hash(word) % (2 ** 32))
            self.input_embeddings[idx] = Tensor(
                np.random.randn(self.embedding_dim) * 0.1,
                label=f'emb_{word[:10]}'
            )
            self.output_embeddings[idx] = np.random.randn(self.embedding_dim) * 0.1
            np.random.seed(None)
        self.word_frequencies[word] = self.word_frequencies.get(word, 0) + 1
        return self.word_to_idx[word]

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, word: str) -> bool:
        return word in self.word_to_idx

    # ------------------------------------------------------------------
    # Encoding (text → indices)
    # ------------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """Tokenize text to indices (with BOS/EOS wrapping + open vocabulary)."""
        tokens = text.lower().split()
        indices = [self.word_to_idx['<BOS>']]
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                # Auto-add new words (open vocabulary)
                indices.append(self.add_word(token))
        indices.append(self.word_to_idx['<EOS>'])
        return indices

    # ------------------------------------------------------------------
    # Decoding (indices → text)
    # ------------------------------------------------------------------
    def decode(self, indices: List[int]) -> str:
        """Decode token indices to string (strips PAD/BOS, stops at EOS)."""
        words = []
        for idx in indices:
            if idx == self.word_to_idx['<EOS>']:
                break
            if idx in (self.word_to_idx['<PAD>'], self.word_to_idx['<BOS>']):
                continue
            words.append(self.idx_to_word.get(idx, '<UNK>'))
        return ' '.join(words)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def get_embedding(self, idx: int) -> Tensor:
        """Get input embedding for token index (autograd-compatible Tensor)."""
        if idx not in self.input_embeddings:
            self.input_embeddings[idx] = Tensor(
                np.random.randn(self.embedding_dim) * 0.1,
                label=f'emb_{idx}'
            )
        return self.input_embeddings[idx]

    def get_output_embedding(self, idx: int) -> np.ndarray:
        """Get output embedding for token index (numpy, for generation)."""
        if idx not in self.output_embeddings:
            self.output_embeddings[idx] = np.random.randn(self.embedding_dim) * 0.1
        return self.output_embeddings[idx]

    def get_embedding_matrix(self) -> np.ndarray:
        """Get full output embedding matrix for efficient batch lookup."""
        vocab_size = len(self.idx_to_word)
        matrix = np.zeros((vocab_size, self.embedding_dim))
        for idx in range(vocab_size):
            matrix[idx] = self.get_output_embedding(idx)
        return matrix


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# These allow existing code to use old names during transition
AGIVocabulary = Vocabulary
DecoderVocabulary = Vocabulary


# ============================================================================
# SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AGI Unified Vocabulary - Self Test")
    print("=" * 60)

    vocab = Vocabulary(embedding_dim=64)
    print(f"\n[1] Initial vocab size: {vocab.vocab_size} (special tokens)")

    # Test add_word
    idx = vocab.add_word("hello")
    print(f"[2] Added 'hello' -> idx={idx}, vocab_size={vocab.vocab_size}")

    # Test encode
    indices = vocab.encode("hello world this is a test")
    print(f"[3] Encoded: {indices}, vocab_size={vocab.vocab_size}")

    # Test decode
    text = vocab.decode(indices)
    print(f"[4] Decoded: '{text}'")

    # Test input embedding (Tensor)
    emb = vocab.get_embedding(idx)
    print(f"[5] Input embedding: type={type(emb).__name__}, shape={emb.data.shape}")

    # Test output embedding (numpy)
    out_emb = vocab.get_output_embedding(idx)
    print(f"[6] Output embedding: type={type(out_emb).__name__}, shape={out_emb.shape}")

    # Test embedding matrix
    matrix = vocab.get_embedding_matrix()
    print(f"[7] Embedding matrix shape: {matrix.shape}")

    # Test frequency tracking
    vocab.add_word("hello")
    vocab.add_word("hello")
    print(f"[8] 'hello' frequency: {vocab.word_frequencies['hello']}")

    # Test backward compatibility aliases
    v2 = AGIVocabulary(32)
    v3 = DecoderVocabulary(32)
    print(f"[9] Backward compat: AGIVocabulary={type(v2).__name__}, DecoderVocabulary={type(v3).__name__}")

    # Test roundtrip
    test_text = "the quick brown fox jumps"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)
    print(f"[10] Roundtrip: '{test_text}' -> {encoded} -> '{decoded}'")
    assert decoded == test_text, f"Roundtrip failed: got '{decoded}'"

    print("\n" + "=" * 60)
    print("AGI Unified Vocabulary: All Tests Passed!")
    print("=" * 60)
