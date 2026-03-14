# Modal x Omniparser

A [Modal](https://modal.com) port of the [OmniParser](https://github.com/microsoft/OmniParser) screen parsing tool.

## Getting Started

```bash
# 1. Install dependencies
uv sync

# 2. Authenticate with modal
uv run modal setup

# 3. Invoke parser on modal
uvx modal run -w response.json parse.py
```

## Formatting

```
uvx ruff format *.py
```

## Building OmniParser

The current image configuration is setup to support [OmniParser v.2.0.1](https://github.com/microsoft/OmniParser/releases/tag/v.2.0.1). The dependencies are pinned in `omniparser-requirements.txt` with tweaks to get the model to actually run. These tweaks include the following:

- `paddleocr<=2.8`
- `torch== 2.2.0`
- `transformers==4.38.2`
- `paddlepaddle<=2.8`
- `cuda-12.2.0`
- `flash_attn-2.6.0+cu122torch2.2cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
