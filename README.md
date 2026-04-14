# EmotionSpeechRecognitionCNN

A convolutional neural network that classifies the **emotional tone of speech** from audio files. The pipeline converts raw `.wav` recordings into mel spectrograms, augments the dataset with pitch-shifting and time-stretching (5× the original data), and trains a 4-layer CNN with global average/max pooling to predict one of eight emotions.

---

## Highlights

- **Full audio-to-prediction pipeline** — raw `.wav` in, emotion label out. Handles spectrogram generation, augmentation, training, and single-file inference.
- **5× data augmentation** — each recording is expanded into five variants: original, pitch-shifted ±2 semitones, and time-stretched at 0.8× / 1.2× rate.
- **Per-actor loudness normalization** — spectrograms are referenced to each actor's mean RMS energy, reducing speaker-level volume bias.
- **GAP + GMP pooling** — global average pooling and global max pooling are concatenated before the classifier head, making the model resolution-independent and capturing both average activation and peak activation cues.
- **Generalization-focused training** — the included checkpoint prioritizes low overfitting over raw accuracy, achieving ~50% test accuracy across 8 classes with the train/validation gap kept small.

---

## Emotions

The model classifies audio into 8 emotion categories (following the RAVDESS labeling scheme):

| Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## Model Architecture

```
Input (3-channel mel spectrogram image)
  │
  ├─ Conv2d(3 → 32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├─ Conv2d(32 → 64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  ├─ Conv2d(64 → 128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → SpatialDropout(0.2)
  ├─ Conv2d(128 → 128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → SpatialDropout(0.2)
  │
  ├─ GlobalAvgPool2d(1) ──┐
  ├─ GlobalMaxPool2d(1) ──┤  concatenate → [256]
  │                        │
  ├─ Linear(256 → 512) → ReLU → Dropout(0.5)
  └─ Linear(512 → 8)
```

All convolutional layers use `padding=1` to preserve spatial dimensions before each pooling step.

---

## Project Structure

```
EmotionSpeechRecognitionCNN/
├── main.ipynb         # Full pipeline: data prep, augmentation, training, evaluation, plots
├── predict.py         # Single-file inference — load audio, generate spectrogram, predict emotion
├── bestModel.pth      # Pre-trained model weights (~1.5 MB)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch (CUDA recommended for training)
- librosa, scikit-learn, torchmetrics, matplotlib, Pillow

```bash
pip install torch torchvision librosa scikit-learn torchmetrics matplotlib pillow
```

### Dataset

The project is designed for the [RAVDESS](https://zenodo.org/record/1188976) dataset — place the audio files under an `archive/` directory at the project root. RAVDESS filenames encode metadata (emotion code at characters 6–7, actor ID at characters 18–19), which the pipeline parses automatically.

### Train

Open and run `main.ipynb` end-to-end. On the first run it will:

1. Compute per-actor mean loudness from the raw audio.
2. Generate mel spectrograms for every file (original + 4 augmented variants) and save them as JPG images organized by emotion folder.
3. Split the spectrogram dataset 80 / 10 / 10 into train, validation, and test sets.
4. Train for 30 epochs, saving the best checkpoint (by validation loss) as `bestModel.pth`.
5. Print accuracy across all three splits, a classification report, and a confusion matrix.

Training hyperparameters:

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 1 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Label smoothing | 0.05 |
| Mixed precision | AMP + GradScaler |
| Epochs | 30 |

### Predict on a Single File

```bash
python predict.py
```

You will be prompted to enter the audio filename (with extension). The script generates a mel spectrogram, runs it through the model, and prints the predicted emotion with a confidence score.

```
Enter name for audio file with extension: sample.wav
prediction : happy, confidence: 72.35%
```

---

## How It Works

### Spectrogram Generation

Each `.wav` file is loaded with librosa at 22 050 Hz. Silent edges are trimmed at a 30 dB threshold. A mel spectrogram is computed and plotted in power-to-dB scale (−80 to 0 dB), referenced to the speaker's mean RMS energy. The plot is saved as a borderless JPG image with no axes, producing a clean image suitable for CNN input.

### Data Augmentation

Every recording yields five training samples: the original audio plus pitch-shifted (+2 and −2 semitones) and time-stretched (0.8× and 1.2× rate) variants. All five are independently converted to spectrograms. This quintuples the dataset and encourages the model to learn emotion-invariant features across pitch and tempo variation.

### Training & Evaluation

The spectrogram images are loaded via `torchvision.datasets.ImageFolder` (3-channel RGB) and split into train/validation/test sets. The model is trained with cross-entropy loss using label smoothing (0.05) and mixed-precision (FP16) training for GPU efficiency. The best checkpoint is selected by validation loss to avoid overfitting.

---

## Results

The included checkpoint achieves roughly **50%+ test accuracy** across 8 emotion classes with minimal overfitting (small gap between train and test accuracy). Higher raw accuracy (~70%+) was achievable but came with significant overfitting, so the generalized model was preferred.

---

## License

No license specified. Contact the repository owner for usage terms.
