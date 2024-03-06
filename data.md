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

print(f"Fit: {p}")
print(f"Compute  cost     C = {C:.3f} core-hour per correlation ({C*60:.3f} core-min)")
print(f"Transfer overhead T = {T:.3f} hour per data stream ({T*3600:.3f} sec)")
print(f"Startup  overhead S = {S:.3f} hour per run ({S*3600:.3f} sec)")
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

## Normalize the Model Constants

Based on the above fitting, assuming the Google Cloud system (CPU, RAM, network, etc) performance are similar to what we will be able to get for a space VLBI missions, we can obtain the cost constants:

```python
bandwidth    = 32 # in Gbps
quantization = 3  # in bit
scanlength   = 20 # in second

comp = C / scanlength / (bandwidth / quantization) # computing cost of correlation in core-hour per second per giga-sample

print(f"Compute cost: comp = {comp:.6f} core-hour per second per giga-sample)")
```

## Data Model for Space VLBI

```python
#obs = eh.obsdata.load_uvfits('m87facing_fullcovonm87.uvfits')
#obs = eh.obsdata.load_uvfits('m87facing_fullcovonsgra.uvfits')
#obs = eh.obsdata.load_uvfits('sgratilted_fullcovonm87.uvfits')
obs = eh.obsdata.load_uvfits('sgratilted_fullcovonsgra.uvfits')

np.unique(np.concatenate([obs.data['t1'], obs.data['t2']]))
```

```python
space='compromi'

S = (obs.data['t1'] == space) | (obs.data['t2'] == space)
G = ~S

U = obs.data['u'] / 1e9
V = obs.data['v'] / 1e9

plt.scatter( U[S],  V[S], s=1, color='b')
plt.scatter(-U[S], -V[S], s=1, color='b')
plt.scatter( U[G],  V[G], s=1, color='g')
plt.scatter(-U[G], -V[G], s=1, color='g')

plt.gca().set_aspect('equal')
```

```python
assert len(np.unique(obs.data['tint'])) == 1

tint = np.unique(obs.data['tint'])[0]

T  = obs.data['time']
TT = np.unique(T)
TT = np.insert(TT, 0, TT[0]-TT[1])
```

```python
Bs = 96  # Gbps
Bg = 192 # Gbps
Qs = 1
Qg = 2

assert Bs/Qs == Bg/Qg

Ds, Dg = [0], [0]
DS, DG = [0], [0]
CA     = [0]

for i, t in enumerate(TT[1:]):
    print(i)
    
    data  = obs.data[T == t]
    array = np.unique(np.concatenate([data['t1'], data['t2']]))

    if space in array:
        Ns = 1
        Ng = len(array) - 1
    else:
        Ns  = 0
        Ng = len(array)

    S  = (data['t1'] == space) | (data['t2'] == space)
    G  = ~S
    NS = np.sum(S)
    NG = np.sum(G)
    assert NS == Ns * Ng # assuming at most one space station
    assert NG == Ng * (Ng - 1) // 2

    Ds.append(Ds[-1] + Ns * Bs * tint)
    Dg.append(Dg[-1] + Ng * Bg * tint)

    CA.append(comp * tint / (TT[i+1]-TT[i]) * (Bg/Qg) * (NS+NG)*(NS+NG-1)/2 / 1e3)

    ################################
    
    #fig, axes = plt.subplots(1,2, figsize=(12,5))

    fig, axd = plt.subplot_mosaic([
        ['l', 'tr'],
        ['l', 'br']
    ], figsize=(12, 5), layout="constrained")
    
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    U = data['u'] / 1e9
    V = data['v'] / 1e9

    axd['l'].scatter( obs.data[T < t]['u']/1e9,  obs.data[T < t]['v']/1e9, s=1, color='k', alpha=0.2)
    axd['l'].scatter(-obs.data[T < t]['u']/1e9, -obs.data[T < t]['v']/1e9, s=1, color='k', alpha=0.2)
    
    axd['l'].scatter( U[S],  V[S], color='b')
    axd['l'].scatter(-U[S], -V[S], color='b')
    axd['l'].scatter( U[G],  V[G], color='g')
    axd['l'].scatter(-U[G], -V[G], color='g')

    axd['l'].set_aspect('equal')
    axd['l'].set_xlim(-25,25)
    axd['l'].set_ylim(-25,25)
    axd['l'].set_xlabel('u (G$\lambda$)')
    axd['l'].set_ylabel('v (G$\lambda$)')

    DS = np.array(Ds) / 8 / 1024 / 1024
    DG = np.array(Dg) / 8 / 1024 / 1024
    N  = len(DS)
    
    axd['tr'].plot(TT[:N], DS,      label='space')
    axd['tr'].plot(TT[:N], DG,      label='ground')
    axd['tr'].plot(TT[:N], DS + DG, label='all')
    axd['tr'].set_xlabel('Time (hr)')
    axd['tr'].set_ylabel('Data Volume (PB)')
    axd['tr'].legend()

    axd['br'].plot(TT[:N], CA)
    axd['br'].set_xlabel('Time (hr)')
    axd['br'].set_ylabel('Cluster size at 1x (1k-core)')

    fig.savefig(f'frames/{i+1:04d}.png')
    plt.close()
```

```python

```
