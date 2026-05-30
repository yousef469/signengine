# SignEngine — Gemma ASL Training Pipeline

Train Gemma 3B to recognize American Sign Language from coordinate sequences. The model reads XYZ hand/face landmark coordinates and outputs sign names with chain-of-thought geometric reasoning.

## How It Works

```
Coordinate Data (XYZ, velocity, face) → Format as conversation → QLoRA fine-tune Gemma → Trained model knows signs
Later: MediaPipe (live camera) → coordinates → Trained Gemma → English text
```

## Project Structure

```
signengine/
├── config.py                        # All settings, vocab, hyperparams
├── engine/
│   ├── normalizer.py                # Shoulder-relative coordinate normalization
│   ├── prompt_builder.py            # Format coordinates → Gemma conversations
│   ├── face_processor.py            # Face mesh derived values + grammar
│   ├── inference.py                 # Load adapter + predict signs
│   ├── segmenter.py                 # Real-time sign boundary detection
│   └── sentence_assembler.py        # Assemble signs → sentences
├── scripts/
│   ├── download_dataset.py          # Fetch/load coordinate data
│   ├── augment.py                   # Speed/mirror/scale/noise augmentation
│   ├── format_training.py           # JSON → training JSONL conversations
│   └── evaluate.py                  # Test accuracy, find confusion pairs
├── training/
│   ├── phase0_proof.py              # Zero-shot test (Ollama, no training)
│   ├── phase1_isolated.py           # QLoRA: single sign recognition
│   ├── phase2_confusion.py          # Confusion pair contrast training
│   ├── phase2b_face.py              # Face grammar + emotion training
│   ├── phase3_idle.py               # Idle/transition handling
│   ├── phase4_sequences.py          # 2-3 sign sequences
│   ├── phase5_continuous.py         # Full continuous sentences
│   └── phase6_corrections.py        # Self-improving correction loop
└── data/
    ├── raw/                         # Raw coordinate JSON from datasets
    ├── processed/                   # Normalized samples
    ├── augmented/                   # After augmentation
    ├── training/                    # JSONL training conversations
    └── corrections/                 # Phase 6 correction queue
```

## Training Phases

| Phase | What It Teaches | Expected Accuracy |
|-------|----------------|-------------------|
| 0 | Proof of concept (zero-shot) | 55-70% |
| 1 | 200 isolated signs | 83-88% |
| 2 | Confusion pair disambiguation | 93-96% |
| 2B | Face grammar + emotions | +4-6% on face signs |
| 3 | Idle/transition handling | <3% false positives |
| 4 | 2-3 sign sequences | 91-94% |
| 5 | Full continuous sentences | 88-93% per sign |
| 6 | Self-improving (ongoing) | 97-99% at 6 months |

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Phase 0 — Test the idea (requires Ollama + Gemma)
```bash
ollama pull gemma3:4b
python training/phase0_proof.py
```

### 3. Add coordinate data
Place coordinate JSON files in `data/raw/{SIGN_NAME}/sample_001.json`

### 4. Process data
```bash
# Augment training data (300 real → 2400 augmented per sign)
python scripts/augment.py --input_dir data/raw --output_dir data/augmented

# Format into training conversations
python scripts/format_training.py --phase 1
```

### 5. Train Phase 1
```bash
python training/phase1_isolated.py --data data/training/phase1_train.jsonl
```

### 6. Evaluate
```bash
python scripts/evaluate.py --adapter models/adapters/phase1 --test data/training/phase1_test.jsonl
```

### 7. Continue phases
```bash
python scripts/format_training.py --phase 2
python training/phase2_confusion.py --adapter models/adapters/phase1
# ... and so on through Phase 6
```

## Coordinate Data Format

Every sample is a JSON file matching this schema:
```json
{
  "sign": "HELLO",
  "source": "wlasl_video_07234",
  "num_frames": 30,
  "fps": 30,
  "frames": [
    {
      "frame_index": 0,
      "timestamp_ms": 0,
      "right_hand": {
        "WRIST": {"x": 0.612, "y": 0.445, "z": 0.023},
        "THUMB_TIP": {"x": 0.634, "y": 0.412, "z": 0.019},
        "INDEX_TIP": {"x": 0.598, "y": 0.389, "z": 0.031}
      },
      "left_hand": { ... },
      "velocity_right": {"x": 0.0, "y": 0.0, "z": 0.0},
      "velocity_left": {"x": 0.0, "y": 0.0, "z": 0.0},
      "pose": {
        "NOSE": {"x": 0.501, "y": 0.312, "z": 0.0},
        "RIGHT_SHOULDER": {"x": 0.620, "y": 0.520, "z": 0.0},
        "LEFT_SHOULDER": {"x": 0.380, "y": 0.520, "z": 0.0}
      }
    }
  ]
}
```

## Integration with SignBridge

Once trained, the adapter plugs into SignBridge:
```python
from engine.inference import SignEngine

engine = SignEngine("models/adapters/phase5")
result = engine.predict(coordinates_from_mediapipe)
print(result["sign"])  # "HELLO"
```
