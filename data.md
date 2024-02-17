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

We start by important some standard python packages.

```python
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np

import ehtim as eh
```

We use benchmark result from cloud correlation and EHT's 2017 data to estimate the data caching, throughput, and computation requirements.

```python
from numpy import nan

s    = np.array([2,    3,    4,    5,    6,    7,    8])
ch8  = np.array([1392, 4918, nan,  nan,  nan,  nan,  nan])  * 8  / 3600
ch16 = np.array([559,  1159, 1977, 3559, 6801, nan,  nan])  * 16 / 3600
ch32 = np.array([287,  491,  815,  1142, 1810, 2102, 2695]) * 32 / 3600
ch64 = np.array([190,  285,  410,  554,  848,  1001, 1346]) * 64 / 3600

nvm  = np.array([3, 4, 3, 1, 1, 1, 3]) # number of VMs used in the benchmarks

#ch8  = np.array([274+1392, nan,      nan,      nan,      nan,      nan,      nan])      * 8  / 3600
#ch16 = np.array([228+559,  295+1159, 306+1977, nan,      nan,      nan,      nan])      * 16 / 3600
#ch32 = np.array([233+287,  238+491,  309+815,  299+1142, 390+1810, 330+2102, 356+2695]) * 32 / 3600
#ch64 = np.array([264+190,  221+285,  298+410,  287+554,  296+848,  312+1001, 314+1346]) * 64 / 3600
```

## Model the Performance

We model the required core-hour as a 2nd-order polynomial,
$$ch = C \frac{s (s - 1)}{2} + c[T s + S]
     = \frac{C}{2} s^2 + \left(cT - \frac{C}{2}\right)s + cS$$
where $ch$ is required core-hour to finish the correlation task, $s$ is the number of stations, and $c$ is the number of cores in the benchmarked VM;
$C$ is the compute cost in core-hour per correlation, which may include multiple polarizations;
$T$ is the data transfer overhead in hour *within a VM*, and $S$ is the startup overhead in hour.

Larger VMs tend to be a bit more efficient in carrying out the correlation (because of how DiFX was designed),
but have less available on Google Data Centers.
For the purpose of this notebook, let's use the 32-core VMs and perform the model fitting using `ch32` data.

```python
c = 32
p = np.polyfit(s, ch32, 2, w=nvm)

C =  p[0] * 2
T = (p[1] + C/2)/c
S =  p[2] / c

print(p)
print("Compute  cost     C = {:.3f} core-hour per correlation ({:.3f} core-min)".format(C, C*60))
print("Transfer overhead T = {:.3f} hour per data stream ({:.3f} sec)".format(T, T*3600))
print("Startup  overhead S = {:.3f} hour per run ({:.3f} sec)".format(S, S*3600))
```

We then overplot the model with the data.

```python
fig, ax = plt.subplots()

x = np.arange(310)/10+1
ones = x**0
ax.loglog(x, p[2]*ones, 'k--', label="0th order term")
ax.loglog(x, p[1]*x,    'k-.', label="1st order term")
ax.loglog(x, p[0]*x*x,  'k:',  label="2nd order term")
ax.loglog(x, p[0]*x*x + p[1]*x + p[2], 'k', label="Model with 32-core VM")

ax.plot(s, ch64, 'ro', markersize=np.sqrt(2*64), label="64-core VM benchmark")
ax.plot(s, ch32, 'go', markersize=np.sqrt(2*32), label="32-core VM benchmark")
ax.plot(s, ch16, 'bo', markersize=np.sqrt(2*16), label="16-core VM benchmark")
ax.plot(s, ch8,  'yo', markersize=np.sqrt(2*8),  label="8-core VM benchmark")

ax.legend(loc="lower right", fontsize=9)
ax.set_title("Computing Cost for Correlating One EHT Scan")
ax.set_xlabel("Number of Stations")
ax.set_ylabel("Measured Cost of 20sec scans [core-hour]")
ax.set_xlim(1, 32)

y0, y1 = ax.get_ylim()

ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(y0 * 15, y1 * 15)
ax2.set_ylabel("Scaled Cost of 5min scans [core-hour]")

fig.savefig("compute.pdf")
```

We can also compare the diffrent components.

```python
fig, ax = plt.subplots()

ax.loglog(x, p[0]*x*x + p[1]*x + p[2], 'k', label="Model with 32-core VM")
ax.loglog(x, c*S*ones,    'k--', label="Overhead: $cS$")
ax.loglog(x, c*T*x,       'k-.', label="Transfer: $cTs$")
ax.loglog(x, C*x*(x-1)/2, 'k:',  label="Compute: $Cs(s-1)/2$")

ax.legend(loc="lower right", fontsize=9)
ax.set_title("Computing Cost for Correlating One EHT Scan")
ax.set_xlabel("Number of Stations")
ax.set_ylabel("Measured Cost of 20sec scans [core-hour]")
ax.set_xlim(1, 32)

y0, y1 = ax.get_ylim()

ax2 = ax.twinx()
ax2.set_yscale("log")
ax2.set_ylim(y0 * 15, y1 * 15)
ax2.set_ylabel("Scaled Cost of 5min scans [core-hour]")

fig.savefig("contributions.pdf")
```
