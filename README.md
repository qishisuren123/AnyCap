# AnyCap: A Unified Framework for Controllable Caption Generation

## Overview
AnyCap is a unified framework for controllable caption generation across different modalities including text, images, and videos. It provides tools for caption generation, model training, evaluation, and fine-tuning. The framework integrates advanced multimodal capabilities with a flexible pipeline, enabling high-quality and customizable captions for a wide variety of input data types.

This repository provides the necessary code for training and evaluating the AnyCap framework, along with dataset pipelines and predefined models. It also offers a custom benchmark (AnyCapEval) to evaluate the model's performance on various multimodal tasks.

## Key Features
- **Multimodal Support**: Generate captions for images, videos, and audio.
- **Customizable Caption Styles**: Control caption styles through predefined instructions and models.
- **Benchmarks and Evaluation**: Evaluate model performance using AnyCapEval, a benchmark designed for multimodal captioning.
- **User-Friendly Pipeline**: Predefined datasets and pipelines for easy training and evaluation.
  
## Installation

### Requirements
To run AnyCap, please install the following dependencies:

```bash
pip install -r requirements.txt
```

You can generate this `requirements.txt` by running `pip freeze > requirements.txt` from your virtual environment.

### Cloning the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/qishisuren123/AnyCap.git
cd AnyCap
```

### Dataset Setup

1. **AnyCapDataset**: Contains datasets for audio, image, and video modalities. The video data is currently empty and can be downloaded from the provided Hugging Face repository.
2. **AnyCapEval**: This contains the benchmark for evaluating the generated captions.
3. **MIA-Bench**: This folder contains the evaluation scripts for benchmarking purposes.

### Downloading Data

For video modality, you can download the dataset from the following Hugging Face repository:

```bash
huggingface-cli login
python download_data.py --dataset anycap_video
```

Ensure that you place the downloaded data into the appropriate directory as outlined in the project structure.

---

## Usage

### 1. Generating Captions

To generate captions for a given modality, use the following script:

```bash
python /path/to/gen/gen_xxx.py
```

Once the captions are generated, save them as `content.jsonl` and `style.jsonl`. These files will be used in the evaluation process.

### 2. Running the Evaluation

Next, set the paths for the generated captions in the `anycapeval_video.sh` script:

```bash
OUTPUT_PATH_CONTENT=/path/to/generated/content.jsonl
OUTPUT_PATH_STYLE=/path/to/generated/style.jsonl
```

Then, start the evaluation process:

```bash
bash anycapeval_video.sh
```

### 3. Benchmarking

To run the benchmarking for the AnyCap framework:

1. Generate the captions using the same script above (`gen_xxx.py`).
2. Add the generated `jsonl` paths to the `--caption_path` argument for evaluation:

```bash
python eval_xxx.py --caption_path /path/to/generated/captions.jsonl
```

---

## Evaluation

The **AnyCapEval** benchmark evaluates model performance on a variety of multimodal tasks including:

- Text-to-image captioning
- Video-to-text captioning
- Style-controlled caption generation

To evaluate, run the following commands:

```bash
python eval/evaluate_model.py --model_path /path/to/your/model
```

The performance metrics and results will be displayed at the end of the evaluation.

---

## Contributing

We welcome contributions to AnyCap! Please feel free to fork the repository, submit issues, or open pull requests.

### Guidelines

- Ensure your code is well-documented.
- Write tests for new features and fixes.
- Follow the style of the existing codebase.

---

## Citation

If you use AnyCap in your research, please cite our paper:

```bibtex
@misc{anycap2025,
  author = {Your Name},
  title = {AnyCap: A Unified Framework for Controllable Caption Generation},
  year = {2025},
  howpublished = {\url{https://github.com/qishisuren123/AnyCap}},
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

