# act.py — Full Function Documentation

**File:** `act.py`  
**Total Lines:** 595  
**Role:** Perfect Reconstruction Engine — reconstructs EXACT original from encoder's latent output  
**Total Functions/Methods:** 13  
**Total Classes:** 5

---

## Class 1: `LatentToNumberRepresentation` (Lines 37–75)

Bridge between AGI semantic latent and ALVS pixel data. Converts latent tensors to/from flat number arrays deterministically.

### Function 1 — `__init__(self, latent_dim=64)` (Line 45)

- Stores configurable `latent_dim` (default 64) for reshaping operations

### Function 2 — `latent_to_numbers(self, latent_z)` (Lines 48–62)

- Accepts latent tensor of shape `(num_slots, latent_dim)`
- Detects if input is a `Tensor` object or raw numpy array
- If `Tensor`: extracts `.data` attribute and flattens to 1D numpy array
- If numpy: directly flattens to 1D
- Returns flat `(num_slots * latent_dim,)` numpy array
- Fully deterministic — no randomness, no loss

### Function 3 — `numbers_to_latent(self, numbers, num_slots)` (Lines 64–75)

- Accepts flat numpy array and `num_slots` count
- Reshapes flat array back to `(num_slots, latent_dim)` using stored `latent_dim`
- Wraps reshaped array into a `Tensor` object
- Returns `Tensor` of shape `(num_slots, latent_dim)`
- Fully deterministic inverse of `latent_to_numbers`

---

## Class 2: `ALVSReconstructionEngine` (Lines 78–241)

Perfect image reconstruction using the ALVS (Atomized Lossless Vision System). Core image reconstruction engine.

### Function 4 — `__init__(self)` (Lines 90–99)

- Instantiates `VisionLoader` for lossless image I/O (JPG/JLS pipeline)
- Instantiates `Atomizer` for creating atomic context (RGB + Energy + Flow)
- Instantiates `Synthesizer` for pixel-perfect image reconstruction
- Instantiates `LatentToNumberRepresentation` with `latent_dim=64`
- Prints initialization status banner with subsystem details

### Function 5 — `load_image_lossless(self, image_path)` (Lines 101–133)

- Takes an image file path as input
- Calls `VisionLoader.load_to_math()` to convert image to lossless math matrix
- Produces `(H, W, 3)` normalized float array (math_matrix)
- Runs `Atomizer.atomize()` on the math matrix to create atomic context (RGB + energy + flow decomposition)
- Returns dictionary containing:
  - `math_matrix`: the normalized float pixel array
  - `atomic_context`: atomized RGB + energy + flow representation
  - `metadata.shape`: original image dimensions
  - `metadata.format`: original file format
  - `metadata.jls_path`: path to JLS (JPEG-LS) intermediate file
- Prints progress logs with checkmarks

### Function 6 — `reconstruct_image_perfect(self, atomic_context, output_path, jls_intermediate=None)` (Lines 135–160)

- Takes atomic context and output path as inputs
- Calls `Synthesizer.reconstruct()` on atomic context to recover the full image matrix
- Calls `VisionLoader.save_from_math()` to write the reconstructed matrix to disk
- Supports optional JLS intermediate path for JPG output pipeline
- Returns the reconstructed image matrix (numpy array)
- Guarantees pixel-perfect / lossless reconstruction

### Function 7 — `encode_with_latent(self, image_path, encoder)` (Lines 162–206)

- Full encoding pipeline: Image → ALVS (pixels) + AGI Encoder (semantics)
- Calls `load_image_lossless()` to get ALVS data
- Imports `SensoryObserver` from `observe` module at call-time
- Converts math_matrix from float64 `[0,1]` to uint8 `[0,255]` for the observer
- Runs `observer.observe()` with `modality='image'`
- Runs `encoder.encode_observation()` to get semantic latent `latent_z`
- Converts `latent_z` to flat number array via `latent_to_numbers()`
- Returns dictionary containing:
  - `latent_z`: semantic latent tensor `(num_slots, latent_dim)`
  - `latent_numbers`: flat numeric representation
  - `atomic_context`: pixel-perfect ALVS data
  - `metadata`: image metadata

### Function 8 — `decode_to_image(self, latent_numbers, atomic_context, output_path, num_slots=6, jls_intermediate=None)` (Lines 208–241)

- **Core act.py role**: reconstructs EXACT image from encoder's latent output
- Converts flat latent_numbers back to latent tensor via `numbers_to_latent()`
- Calls `reconstruct_image_perfect()` using ALVS atomic context
- Accepts configurable `num_slots` (default 6) and optional JLS intermediate path
- Returns reconstructed image array (EXACT original, lossless)
- Future-proofed: designed for latent-guided synthesis/generation

---

## Class 3: `TextReconstructionEngine` (Lines 244–299)

Perfect text reconstruction using deterministic token mapping.

### Function 9 — `__init__(self, latent_dim=64)` (Lines 247–249)

- Stores `latent_dim` parameter
- Instantiates `LatentToNumberRepresentation` with given latent_dim

### Function 10 — `encode_text(self, text, encoder)` (Lines 251–278)

- Imports `SensoryObserver` from `observe` at call-time
- Runs `observer.observe()` with `modality='text'` on input string
- Runs `encoder.encode_observation()` to get semantic latent
- Converts latent to flat numbers via `latent_to_numbers()`
- Stores the original text and original token observation for perfect reconstruction
- Returns dictionary with `latent_numbers`, `original_text`, and `original_tokens`

### Function 11 — `decode_text(self, latent_numbers, original_tokens, num_slots=6)` (Lines 280–299)

- Restores latent tensor from flat numbers via `numbers_to_latent()`
- Uses stored original tokens for perfect (lossless) text reconstruction
- No information loss — direct token replay
- Future-proofed: latent can be used for generation/variation in future

---

## Class 4: `AudioReconstructionEngine` (Lines 302–534)

Perfect audio reconstruction using NASS (Nexuss Audio Substrate System).

### Function 12 — `__init__(self, latent_dim=64)` (Lines 313–341)

- Stores `latent_dim` and creates `LatentToNumberRepresentation`
- Dynamically adds NASS path to `sys.path`
- Imports NASS components at init-time with try/except:
  - `waveform_to_tensor` and `tensor_to_waveform` from `core.math_kernel`
  - `AudioReader` and `AudioWriter` from `core.io_stream`
  - `config` as `nass_config`
- Sets `nass_available` flag based on import success
- On success: prints NASS info (precision, sample rate, FFT size)
- On failure: sets `nass_available = False` and prints fallback notice
- NASS features: Float16 precision, lossless STFT/iSTFT, multi-quality output, parallel processing

### Function 13 — `encode_audio(self, audio_path, encoder)` (Lines 343–407)

- Full audio encoding: Audio file → NASS tensor + AGI semantic latent
- Imports `SensoryObserver` at call-time
- Runs `observer.observe()` with `modality='audio'`
- Runs `encoder.encode_observation()` for semantic latent
- Converts latent to flat numbers
- If NASS available:
  - Casts waveform to NASS dtype (Float16) for 50% memory reduction
  - Transforms waveform to frequency domain via `waveform_to_tensor()` (lossless STFT)
  - Stores complex tensor shape and dtype metadata
- If NASS unavailable: sets `nass_data = None` (fallback mode)
- Returns dictionary with:
  - `latent_numbers`: AGI semantic encoding
  - `nass_tensor`: frequency-domain complex tensor (or None)
  - `original_waveform`: raw time-domain waveform (fallback safety net)
  - `metadata`: sample_rate, duration, waveform shape

### Function 14 — `decode_audio(self, latent_numbers, nass_tensor, original_waveform, output_path, num_slots=6, quality='360k')` (Lines 409–498)

- Restores latent tensor from flat numbers
- **NASS reconstruction path** (when available):
  - Performs inverse STFT via `tensor_to_waveform()` — lossless frequency→time domain
  - **Multi-format output pipeline:**
    1. Always constructs M4A first (lossless intermediate via `AudioWriter` with `quality='lossless'`)
    2. Reads config flags from `nass_config`:
       - `ENABLE_MP3_360K` (default True) → produces 360kbps MP3
       - `ENABLE_MP3_128K` (default False) → produces 128kbps MP3
       - `ENABLE_M4A_LOSSLESS` (default False) → keeps/deletes M4A intermediate
    3. Creates MP3 files at configured bitrates using `AudioWriter`
    4. Deletes M4A intermediate if `ENABLE_M4A_LOSSLESS` is False
  - Output filenames derived from `output_path` base name with suffixes
- **Fallback path** (no NASS):
  - Saves original waveform to output_path using `soundfile.write()` at 22050 Hz
  - Wrapped in try/except for graceful error handling
- Returns reconstructed waveform numpy array

### Function 15 — `convert_audio_quality(self, input_path, output_path, quality='360k')` (Lines 500–534)

- Standalone audio quality converter using NASS pipeline
- Checks `nass_available` and returns early if unavailable
- Creates `AudioReader` and `AudioWriter` for streaming conversion
- Processes audio in chunks via `reader.stream_chunks()`
- For each chunk:
  - Transforms to frequency domain (`waveform_to_tensor`)
  - Transforms back to time domain (`tensor_to_waveform`) — lossless round-trip
  - Writes reconstructed chunk to output
- Supports quality modes: `'360k'`, `'128k'`, `'lossless'`
- Streaming architecture — handles arbitrary-length audio without full memory load

---

## Class 5: `PerfectReconstructionEngine` (Lines 541–559)

Unified interface for perfect reconstruction across ALL modalities.

### Function 16 — `__init__(self, latent_dim=64)` (Lines 550–559)

- Instantiates `ALVSReconstructionEngine` for image reconstruction
- Instantiates `TextReconstructionEngine` with given latent_dim
- Instantiates `AudioReconstructionEngine` with given latent_dim
- Prints initialization summary: Image (ALVS), Text (Token storage), Audio (Waveform storage), Latent (384 dims)
- Acts as the single entry point for all reconstruction tasks

---

## `__main__` Block (Lines 566–595)

- Runs ALVS integration test when `act.py` is executed directly
- Instantiates `ALVSReconstructionEngine`
- Tests with sample image `Image-text/samples/dog.jpg` (if file exists)
- Calls `load_image_lossless()` → `reconstruct_image_perfect()`
- Validates reconstruction by computing max pixel difference between original and reconstructed
- Prints original shape, reconstructed shape, and pixel difference metric

---

## Summary Table

| # | Class | Method | Lines | Purpose |
|---|-------|--------|-------|---------|
| 1 | `LatentToNumberRepresentation` | `__init__` | 45–46 | Store latent_dim config |
| 2 | `LatentToNumberRepresentation` | `latent_to_numbers` | 48–62 | Tensor → flat numpy (deterministic) |
| 3 | `LatentToNumberRepresentation` | `numbers_to_latent` | 64–75 | Flat numpy → Tensor (deterministic inverse) |
| 4 | `ALVSReconstructionEngine` | `__init__` | 90–99 | Init VisionLoader, Atomizer, Synthesizer |
| 5 | `ALVSReconstructionEngine` | `load_image_lossless` | 101–133 | Image → math matrix + atomic context |
| 6 | `ALVSReconstructionEngine` | `reconstruct_image_perfect` | 135–160 | Atomic context → pixel-perfect image |
| 7 | `ALVSReconstructionEngine` | `encode_with_latent` | 162–206 | Image → ALVS data + AGI latent |
| 8 | `ALVSReconstructionEngine` | `decode_to_image` | 208–241 | Latent + ALVS → exact image |
| 9 | `TextReconstructionEngine` | `__init__` | 247–249 | Store latent_dim, create converter |
| 10 | `TextReconstructionEngine` | `encode_text` | 251–278 | Text → latent + stored tokens |
| 11 | `TextReconstructionEngine` | `decode_text` | 280–299 | Stored tokens → perfect text |
| 12 | `AudioReconstructionEngine` | `__init__` | 313–341 | Init NASS integration with fallback |
| 13 | `AudioReconstructionEngine` | `encode_audio` | 343–407 | Audio → NASS tensor + AGI latent |
| 14 | `AudioReconstructionEngine` | `decode_audio` | 409–498 | NASS tensor → multi-format audio output |
| 15 | `AudioReconstructionEngine` | `convert_audio_quality` | 500–534 | Streaming quality conversion |
| 16 | `PerfectReconstructionEngine` | `__init__` | 550–559 | Unified multi-modal interface |
