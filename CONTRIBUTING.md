# Contributing to Subhadra

Thank you for your interest in contributing to Subhadra — a native Odia AI project. This document explains how to get involved.

---

## Ways to Contribute

### 1. Odia Language Data
The biggest bottleneck is high-quality Odia data. You can help by:
- Adding more Q&A pairs to `data/build_sft_data.py`
- Contributing Odia text corpora (Wikipedia articles, books, news)
- Recording Odia speech for TTS training (see below)

### 2. Model Improvements
- Improve the SFT dataset quality and diversity
- Experiment with larger model configs in `config.yaml`
- Improve the generation quality (better sampling, beam search)

### 3. Bug Fixes
- Check open issues on GitHub
- Fix errors in training scripts, inference, or API

### 4. Documentation
- Improve this README or CONTRIBUTING guide
- Add docstrings to undocumented functions
- Write tutorials or blog posts about the project

### 5. Female TTS Voice (High Priority)
We need a native Odia female speaker to record ~1-2 hours of speech for training a custom female TTS voice. If you or someone you know speaks Odia fluently and is willing to help, please open an issue.

---

## Getting Started

### Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/Subhadra-AI.git
cd Subhadra-AI
pip install -r requirements.txt
```

### Create a branch

```bash
git checkout -b feature/your-feature-name
```

### Make your changes, then push

```bash
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

---

## Adding SFT Data

The SFT dataset is in `data/build_sft_data.py` as a list of `(question, answer)` tuples in Odia.

To add new pairs:

```python
# In data/build_sft_data.py, add to QA_PAIRS:
("ଆପଣଙ୍କ ପ୍ରଶ୍ନ?", "ଆପଣଙ୍କ ଉତ୍ତର।"),
```

Then regenerate the dataset:

```bash
python data/build_sft_data.py
```

Guidelines for good SFT pairs:
- Questions should be natural Odia, not translated from English
- Answers should be factually correct and concise
- Cover diverse topics: culture, history, science, daily life, Odisha
- Avoid repetitive or near-duplicate pairs

---

## Recording TTS Data

If you want to contribute a female Odia voice:

1. Read sentences from `data/clean/odia_corpus.txt`
2. Record in a quiet room using a decent microphone
3. Save as `.wav` files, 22050 Hz, mono
4. Each clip should be 5-15 words, 2-8 seconds long
5. Aim for 1000+ clips minimum

Open an issue titled "TTS Voice Contribution" and we'll coordinate from there.

---

## Code Style

- Python 3.10+
- Follow existing code style (no strict linter enforced yet)
- Add docstrings to new functions
- Keep functions focused and small
- No external dependencies without discussion

---

## Commit Message Format

```
feat: add new feature
fix: fix a bug
docs: update documentation
data: add or update training data
tune: adjust hyperparameters
refactor: code cleanup
```

---

## Opening Issues

When reporting a bug, include:
- Python version
- GPU/CPU info
- Full error traceback
- Steps to reproduce

When requesting a feature, explain:
- What you want to add
- Why it's useful for Odia AI
- Any relevant references or datasets

---

## Code of Conduct

- Be respectful and welcoming
- Focus on constructive feedback
- This project is for the Odia language community — keep that mission central

---

## Questions?

Open an issue or start a discussion on GitHub. We're happy to help new contributors get started.
