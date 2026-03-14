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
