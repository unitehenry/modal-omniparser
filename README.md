# Modal x Omniparser

<https://github.com/user-attachments/assets/18f41b9e-65fe-4115-acd9-5cc4ac6b2f7b>

An [OmniParser](https://github.com/microsoft/OmniParser) port for the [Modal](https://modal.com) platform.

## Motivation

Downloading and building OmniParser from source is not trivial to do out of the box. This project, and the Modal platform, make it extremely easy for anyone who wants to just pass a screenshot into the model and have it return the parsed content.

## Getting Started

```bash
# 1. Authenticate with modal
uvx modal setup

# 2. Invoke parser on modal
uvx modal run -w response.json parse.py --file-url=https://raw.githubusercontent.com/microsoft/OmniParser/refs/heads/master/imgs/word.png
```

## Formatting

```bash
uvx ruff format *.py
```

## Annotated Images

![annotated website](https://repository-images.githubusercontent.com/1182066345/978ba023-614e-4f49-ac2d-c6779b1a6b68)

Annotated images are stored in a Modal volume called `omniparser` by default. The images can be found under the `omniparser/output` directory.

## Building OmniParser

The current image configuration is setup to support [OmniParser v.2.0.1](https://github.com/microsoft/OmniParser/releases/tag/v.2.0.1). The dependencies are pinned in `omniparser-requirements.txt` with tweaks to get the model to actually run. These tweaks include the following:

- `paddleocr<=2.8`
- `torch==2.2.0`
- `transformers==4.38.2`
- `paddlepaddle<=2.8`
- `cuda-12.2.0`
- `flash_attn-2.6.0+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
