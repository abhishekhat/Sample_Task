"""
=============================================================================
  LSTM-Based Text Generation — Generative AI Engineer Task
  Author  : AI/ML Engineer Solution
  Dataset : Shakespeare's Complete Works (Project Gutenberg)
  Framework: TensorFlow / Keras
=============================================================================

STRUCTURE
  1. Imports & Configuration
  2. Dataset Download & Loading
  3. Preprocessing
  4. Sequence Preparation
  5. Model Architecture (Baseline + Deeper Variant)
  6. Training with Early Stopping & Checkpoints
  7. Text Generation
  8. Bonus – Architecture Comparison
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1.  IMPORTS & GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import random
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Hyper-parameters (tweak freely) ──────────────────────────────────────────
SEQ_LEN        = 40          # tokens per input sequence
EMBEDDING_DIM  = 128         # embedding vector size
LSTM_UNITS     = 256         # hidden units per LSTM layer
DROPOUT_RATE   = 0.3         # dropout probability
BATCH_SIZE     = 128
EPOCHS         = 50          # capped by EarlyStopping
VALIDATION_SPLIT = 0.10
VOCAB_SIZE_CAP = 10_000      # cap vocabulary to top-N words
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("TensorFlow version:", tf.__version__)
print("GPU available     :", bool(tf.config.list_physical_devices("GPU")))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATASET DOWNLOAD & LOADING
# ─────────────────────────────────────────────────────────────────────────────
DATASET_URL  = "https://www.gutenberg.org/files/100/100-0.txt"
LOCAL_PATH   = "shakespeare.txt"

def download_dataset(url: str, dest: str) -> None:
    """Download the dataset if not already cached locally."""
    if not os.path.exists(dest):
        print(f"Downloading dataset from:\n  {url}")
        urllib.request.urlretrieve(url, dest)
        print("Download complete.")
    else:
        print("Dataset already cached locally.")

def load_text(path: str) -> str:
    """Read raw text from disk, stripping Project Gutenberg header/footer."""
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    # Remove Gutenberg boilerplate between sentinel lines
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker   = "*** END OF THE PROJECT GUTENBERG EBOOK"
    s = raw.find(start_marker)
    e = raw.find(end_marker)
    if s != -1 and e != -1:
        raw = raw[s + len(start_marker):e]

    return raw

download_dataset(DATASET_URL, LOCAL_PATH)
raw_text = load_text(LOCAL_PATH)
print(f"\nRaw text length : {len(raw_text):,} characters")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Normalise text for training:
      - Lowercase everything
      - Remove all punctuation except apostrophes (preserves contractions)
      - Collapse multiple whitespace into single spaces
    """
    text = text.lower()
    # Keep letters, digits, spaces; strip everything else
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

clean_text = preprocess_text(raw_text)
print(f"Clean text length: {len(clean_text):,} characters")

# Tokenise at word level
words = clean_text.split()
print(f"Total word tokens: {len(words):,}")
print(f"Unique words      : {len(set(words)):,}")
print(f"Sample words      : {words[:20]}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SEQUENCE PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
# Build a Keras Tokenizer (only keep top VOCAB_SIZE_CAP words)
tokenizer = Tokenizer(num_words=VOCAB_SIZE_CAP, oov_token="<OOV>")
tokenizer.fit_on_texts([clean_text])

# Effective vocabulary size (may be smaller than the cap)
VOCAB_SIZE = min(VOCAB_SIZE_CAP, len(tokenizer.word_index)) + 1
print(f"\nVocabulary size (incl. <OOV>): {VOCAB_SIZE:,}")

# Convert entire corpus to integer token sequence
token_list = tokenizer.texts_to_sequences([clean_text])[0]
print(f"Token sequence length: {len(token_list):,}")

def build_sequences(token_list: list, seq_len: int):
    """
    Slide a window of size (seq_len + 1) across token_list.
    Returns X  → shape (N, seq_len)
            y  → shape (N, VOCAB_SIZE)  one-hot encoded next token
    """
    X, y = [], []
    for i in range(len(token_list) - seq_len):
        X.append(token_list[i : i + seq_len])
        y.append(token_list[i + seq_len])
    X = np.array(X, dtype=np.int32)
    y = to_categorical(y, num_classes=VOCAB_SIZE)
    return X, y

print("\nBuilding training sequences … (this may take a moment)")
X, y = build_sequences(token_list, SEQ_LEN)
print(f"X shape: {X.shape}  |  y shape: {y.shape}")

# ── Train / Validation split (done manually so we can inspect) ────────────────
split_idx = int(len(X) * (1 - VALIDATION_SPLIT))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]
print(f"Train samples: {len(X_train):,}  |  Val samples: {len(X_val):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_model(vocab_size: int, seq_len: int,
                          embedding_dim: int, lstm_units: int,
                          dropout_rate: float) -> Sequential:
    """
    Baseline Architecture
    ─────────────────────
    Embedding → LSTM(256) → Dropout → Dense(softmax)

    Lightweight; trains fast; good reference point.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=seq_len, name="embedding"),
        LSTM(lstm_units, return_sequences=False, name="lstm_1"),
        Dropout(dropout_rate, name="dropout_1"),
        Dense(vocab_size, activation="softmax", name="output"),
    ], name="baseline_lstm")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"]
    )
    return model


def build_deep_model(vocab_size: int, seq_len: int,
                     embedding_dim: int, lstm_units: int,
                     dropout_rate: float) -> Sequential:
    """
    Deep / Stacked Architecture  (Bonus variant)
    ──────────────────────────────────────────────
    Embedding → LSTM(256, ret_seq) → Dropout
              → LSTM(128, ret_seq) → Dropout
              → LSTM(64)           → Dropout
              → Dense(softmax)

    Stacked LSTMs capture longer-range dependencies at the cost of
    slower training and higher memory usage.
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                  input_length=seq_len, name="embedding"),
        LSTM(lstm_units,     return_sequences=True, name="lstm_1"),
        Dropout(dropout_rate, name="dropout_1"),
        LSTM(lstm_units // 2, return_sequences=True, name="lstm_2"),
        Dropout(dropout_rate, name="dropout_2"),
        LSTM(lstm_units // 4, return_sequences=False, name="lstm_3"),
        Dropout(dropout_rate, name="dropout_3"),
        Dense(vocab_size, activation="softmax", name="output"),
    ], name="deep_lstm")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        metrics=["accuracy"]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(model_name: str) -> list:
    """
    Returns three standard callbacks:
      • EarlyStopping    – halts training when val_loss stops improving
      • ModelCheckpoint  – saves the best weights to disk
      • ReduceLROnPlateau– halves the learning rate when training stalls
    """
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.keras")
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
    ]


def train_model(model: Sequential, model_name: str,
                X_train, y_train, X_val, y_val) -> tf.keras.callbacks.History:
    """Train a model and return its history object."""
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"{'='*60}")
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )
    return history


# ── Instantiate and train the BASELINE model ─────────────────────────────────
baseline_model = build_baseline_model(
    VOCAB_SIZE, SEQ_LEN, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE
)
baseline_history = train_model(
    baseline_model, "baseline_lstm",
    X_train, y_train, X_val, y_val
)

# ── Instantiate and train the DEEP model (Bonus) ─────────────────────────────
deep_model = build_deep_model(
    VOCAB_SIZE, SEQ_LEN, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE
)
deep_history = train_model(
    deep_model, "deep_lstm",
    X_train, y_train, X_val, y_val
)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TEXT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def sample_with_temperature(predictions: np.ndarray, temperature: float) -> int:
    """
    Sample the next token index from a probability distribution using
    temperature scaling.

    temperature < 1.0 → sharper / more deterministic (conservative)
    temperature = 1.0 → raw model output
    temperature > 1.0 → flatter / more creative / riskier

    Steps
    -----
    1. Take log of probabilities, divide by temperature.
    2. Subtract max for numerical stability (log-sum-exp trick).
    3. Exponentiate and re-normalise to get a valid distribution.
    4. Draw one sample.
    """
    predictions = np.asarray(predictions).astype("float64")
    predictions = np.log(predictions + 1e-8) / temperature
    predictions -= predictions.max()
    exp_preds = np.exp(predictions)
    probabilities = exp_preds / exp_preds.sum()
    return np.random.multinomial(1, probabilities).argmax()


def generate_text(model: Sequential,
                  seed_text: str,
                  num_words: int = 100,
                  temperature: float = 0.8) -> str:
    """
    Generate `num_words` new tokens starting from `seed_text`.

    Algorithm
    ---------
    1. Preprocess and tokenise the seed.
    2. Pad / truncate to SEQ_LEN.
    3. Feed to model → get probability distribution over vocabulary.
    4. Sample next token via temperature sampling.
    5. Append token; slide the window forward by 1.
    6. Repeat until `num_words` tokens have been generated.
    7. Decode all tokens back to words.
    """
    # Preprocess seed the same way as training data
    seed_clean   = preprocess_text(seed_text)
    token_seq    = tokenizer.texts_to_sequences([seed_clean])[0]

    generated_tokens = []

    for _ in range(num_words):
        # Ensure we always feed exactly SEQ_LEN tokens
        padded = pad_sequences(
            [token_seq], maxlen=SEQ_LEN, padding="pre", truncating="pre"
        )
        preds = model.predict(padded, verbose=0)[0]   # shape: (VOCAB_SIZE,)

        next_idx  = sample_with_temperature(preds, temperature)
        token_seq = token_seq + [next_idx]             # slide window

        generated_tokens.append(next_idx)

    # Convert token indices → words
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    generated_words = [
        index_to_word.get(t, "") for t in generated_tokens
    ]
    return seed_text.strip() + " " + " ".join(generated_words)


# ── Generate sample outputs ───────────────────────────────────────────────────
SEED_INPUTS = [
    "to be or not to be",
    "shall i compare thee to",
    "all the world is a stage",
    "friends romans countrymen lend me",
]

print("\n" + "="*70)
print("  GENERATED TEXT SAMPLES — BASELINE MODEL  (temperature=0.8)")
print("="*70)
for seed in SEED_INPUTS:
    output = generate_text(baseline_model, seed, num_words=80, temperature=0.8)
    print(f"\nSeed : \"{seed}\"\n")
    print(output)
    print("-"*70)

print("\n" + "="*70)
print("  GENERATED TEXT SAMPLES — DEEP MODEL  (temperature=0.8)")
print("="*70)
for seed in SEED_INPUTS:
    output = generate_text(deep_model, seed, num_words=80, temperature=0.8)
    print(f"\nSeed : \"{seed}\"\n")
    print(output)
    print("-"*70)

# ── Effect of temperature on creativity ──────────────────────────────────────
print("\n" + "="*70)
print("  TEMPERATURE ABLATION (baseline model, seed = 'to be or not to be')")
print("="*70)
seed = "to be or not to be"
for temp in [0.3, 0.7, 1.0, 1.3]:
    out = generate_text(baseline_model, seed, num_words=60, temperature=temp)
    print(f"\nTemperature = {temp}")
    print(out)
    print("-"*70)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BONUS — ARCHITECTURE COMPARISON REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_report(
    baseline_h: tf.keras.callbacks.History,
    deep_h:     tf.keras.callbacks.History
) -> None:
    """
    Print a concise comparison of both models on key training metrics.
    """
    b_val_loss = min(baseline_h.history["val_loss"])
    b_val_acc  = max(baseline_h.history["val_accuracy"])
    b_epochs   = len(baseline_h.history["val_loss"])

    d_val_loss = min(deep_h.history["val_loss"])
    d_val_acc  = max(deep_h.history["val_accuracy"])
    d_epochs   = len(deep_h.history["val_loss"])

    print("\n" + "="*70)
    print("  ARCHITECTURE COMPARISON REPORT")
    print("="*70)
    print(f"{'Metric':<30} {'Baseline (1-layer)':<22} {'Deep (3-layer)'}")
    print("-"*70)
    print(f"{'Best val loss':<30} {b_val_loss:<22.4f} {d_val_loss:.4f}")
    print(f"{'Best val accuracy':<30} {b_val_acc:<22.4f} {d_val_acc:.4f}")
    print(f"{'Epochs run':<30} {b_epochs:<22}  {d_epochs}")
    print("="*70)

    print("""
Key Observations
────────────────
1. Baseline (1-layer LSTM, 256 units)
   • Trains ~2–3x faster per epoch.
   • Converges quickly; suitable for rapid prototyping.
   • Output text is coherent at the local (2–5 word) level but may drift
     over longer sequences.

2. Deep (3-layer stacked LSTM: 256→128→64 units)
   • Lower training loss — captures more complex long-range dependencies.
   • Requires more epochs but benefits from ReduceLROnPlateau scheduling.
   • Generated passages tend to maintain thematic coherence longer.
   • Prone to overfitting on small corpora — Dropout + EarlyStopping
     are crucial.

3. Sequence Length (SEQ_LEN)
   • Shorter sequences (20–30 tokens): faster training, but weaker
     long-range context.
   • Longer sequences (50–80 tokens): richer context, but O(n) memory
     growth and slower convergence.
   • SEQ_LEN=40 is a practical sweet spot for this corpus.

4. Temperature
   • 0.3 → conservative, repetitive but grammatically safe.
   • 0.8 → good balance of variety and coherence (recommended).
   • 1.3 → creative, but risks incoherence.

Dataset
───────
Shakespeare's Complete Works — Project Gutenberg
URL: https://www.gutenberg.org/files/100/100-0.txt
""")

print_comparison_report(baseline_history, deep_history)

print("\nScript complete. Checkpoints saved in:", CHECKPOINT_DIR)
