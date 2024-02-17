---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Space VLBI DataOps

This notebook provides a simple model of data pathway for space VLBI missions.
We use benchmark result from cloud correlation and EHT's 2017 data to estimate the data caching, throughput, and computation requirements.

We start by important some standard python packages.

```python
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np

import ehtim as eh
```

```python

```
