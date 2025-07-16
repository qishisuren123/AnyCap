# AnyCapEval Benchmark

A unified, multi-modal evaluation benchmark for **controllable captioning** across images, videos, and audio.
AnyCapEval is designed to test both content adherence (how well captions follow explicit user instructions)
and style consistency (fluency, tone, and expressiveness) under a diversity of control directives.

## Repository Structure

```
AnyCapEval/
├── anycapeval_image/    # Test examples for image captioning (instruction, reference, candidate)
├── anycapeval_video/    # Test examples for video captioning
├── anycapeval_audio/    # Test examples for audio captioning
└── LICENSE              # Apache-2.0 license for data
```

## Dataset Description

- **Modalities:** Image, Video, Audio  
- **Examples:** Each example is a triplet `(instruction, high_quality_caption, low_quality_caption)`  
- **Evaluation Dimensions:**  
  - **Content:** measured via keypoint density and human/GPT-based content judgments  
  - **Style:** scored on a 0–4 rubric for narrative, poetic, brief, and detailed captions  

## Quick Start

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("qishisuren/AnyCapEval", split="test")
print(ds[0])
```

## License

This dataset is released under the **Apache-2.0** license. See `LICENSE` for details.
