#source: https://github.com/giopaglia/rooflini/blob/master/rooflini.py
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker

import numpy as np
import sys
import pylab
import re
import json
import math

##########################################################
######################## Settings ########################
##########################################################

# Filename
filename = "roofline.pdf"
if len(sys.argv) > 1:
  filename = sys.argv[1]

# Axis limits
xmin, xmax, ymin, ymax = 0.04, 10000, 0.4, 30000000

# Figure
fig_ratio = 2
fig_dimension = 7
fig = plt.figure()
ax = plt.subplot(1,1,1)
ax.grid(color="#DDDDDD", zorder=-1)
ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]", fontsize=15)
ax.set_ylabel("Performance [GFLOP/s]", fontsize=15)

##########################################################
########################## Data ##########################
##########################################################

# Architecture-specific roofs
# Source: https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf#%5B%7B%22num%22%3A49%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22XYZ%22%7D%2C70%2C450%2C0%5D
# https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet
gpu_roofs = [
#   {"name" : "Scalar Add Peak",    "val" : 98.48},
#   {"name" : "DP Vector Add Peak", "val" : 843.06},
#   {"name" : "DP Vector FMA Peak", "val" : 1691.96},
  # {"name" : "(H100) Peak 16-Bit Floating Point Performance with Tensor Cores (assuming sparsity)", "val" : 1979000},
  {"name" : "(H100) Peak 16-Bit Floating Point Performance with Tensor Cores (no sparsity)", "val" : 1000000},
  # {"name" : "(H100) Peak 16-Bit Floating Point Performance with Streaming Multiprocessors", "val" : 120000},
  {"name" : "rtx 3090ti Peak 16-Bit Floating Point Performance with Tensor Cores", "val" : 79994.88},
]

# Memory in Giga Bytes Per Second
mem_bottlenecks = [
    # {"name" : "L1 Bandwidth",     "val" : 7398.95},
    # {"name" : "L2 Bandwidth",     "val" : 1237.34},
    # {"name" : "MCDRAM Bandwidth", "val" : 393.75},
    {"name" : "Nvidia 6000",    "val" : 1790},
    {"name" : "HBM2e (H100)",    "val" : 3350}, 
    {"name" : "RTX 3090ti",    "val" : 3350}
]

# Benchmarks
AI_v = {
  # "(Layer Norm Fwd Quant) Batch Size 1, Sequence Length 161, New Tokens 1" : 6.882442748091603,
  # "(Layer Norm Fwd Quant) Batch Size 256, Sequence Length 161, New Tokens 1" : 3.5191431943031537,
  "Batch Size 1" : 94.19251092436974,
  "(Effective) Batch Size 1" : 544.752911518972, 
  "Batch Size 8": 355.0904576802508, 
  "(Effective) Batch Size 8" : 883.144666691159, 
  "Batch Size 1024": 360.18445040214476,
  "(Effective) Batch Size 1024" : 365.486673150519

  # "(GEMM Kernels) Batch Size 256, Sequence Length 161, New Tokens 1" : 0.1269359383082901,

  # "(Sigmoid Kernels) Batch Size 1, Sequence Length 161, New Tokens 1" : 5.414220545033318,
  # "(Sigmoid Kernels) Batch Size 256, Sequence Length 161, New Tokens 1" : 3.4061410418195157,

  # "(Swiglu Kernels) Batch Size 1, Sequence Length 161, New Tokens 1" : 4.299020971302428,
  # "(Swiglu Kernels) Batch Size 256, Sequence Length 161, New Tokens 1" : 2.7353975077773938,

  # "(Fused Recurrent HGRN Kernels) Batch Size 1, Sequence Length 161, New Tokens 1" : 1.4959036144578313,
  # "(Fused Recurrent HGRN Kernels) Batch Size 256, Sequence Length 161, New Tokens 1" : 0.5391829997797021
}

# Datapoints
datapoints = [
  # {"AI" : "(Layer Norm Fwd Quant) Batch Size 1, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 1012.3228070175439,  "label"  : "Layer Norm Small Batch"},
  # {"AI" : "(Layer Norm Fwd Quant) Batch Size 256, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 5020.780493468795,  "label"  : "Layer Norm Large Batch"},

  {"AI" : "Batch Size 1", "GFLOP/s" : 47043.54560716284,  "label"  : "Batch Size 1"}, 
  {"AI" : "(Effective) Batch Size 1", "GFLOP/s" : 47043.54560716284,  "label"  : "(Effective) Batch Size 1"},
  {"AI" : "Batch Size 8", "GFLOP/s" : 52998.57577317175,  "label"  : "Batch Size 8"}, 
  {"AI" : "(Effective) Batch Size 8", "GFLOP/s" : 52998.57577317175,  "label"  : "(Effective) Batch Size 8"},
  {"AI" : "Batch Size 1024", "GFLOP/s" : 74638.22222222223,  "label"  : "Batch Size 1024"}, 
  {"AI" : "(Effective) Batch Size 1024", "GFLOP/s" : 74638.22222222223,  "label"  : "(Effective) Batch Size 1024"},
  # {"AI" : "(GEMM Kernels) Batch Size 256, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 62.066447058823535,  "label"  : "GEMM Large Batch"},

  # {"AI" : "(Sigmoid Kernels) Batch Size 1, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 1025.737556561086,  "label"  : "Sigmoid Small Batch"},
  # {"AI" : "(Sigmoid Kernels) Batch Size 256, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 4999.321847002068,  "label"  : "Sigmoid Large Batch"},

  # {"AI" : "(Swiglu Kernels) Batch Size 1, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 1288.0085416666668,  "label"  : "Swiglu Small Batch"},
  # {"AI" : "(Swiglu Kernels) Batch Size 256, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 4052.700171561725,  "label"  : "Swiglu Large Batch"},

  # {"AI" : "(Fused Recurrent HGRN Kernels) Batch Size 1, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 30.9394467979068,  "label"  : "Fused Recurrent HGRN Small Batch"},
  # {"AI" : "(Fused Recurrent HGRN Kernels) Batch Size 256, Sequence Length 161, New Tokens 1",        "GFLOP/s" : 707.7841348934467,  "label"  : "Fused Recurrent HGRN Large Batch"}

  #{"AI" : 10,                        "GFLOPs" : 1000.00,  "label"  : "I'm pretty cool",  "prop" : ["whatever", "..."]},
  #{"AI" : "MyCPUIntensiveBenchmark", "GFLOPs" : 3000.00, "label"  : "I'm unbelievable", "prop" : ["whatever", "..."]},

]

##########################################################
# set_size for explicitly setting axes widths/heights
# see: https://stackoverflow.com/a/44971177/5646732

def set_size(w,h, ax=None):
  """ w, h: width, height in inches """
  if not ax: ax=plt.gca()
  l = ax.figure.subplotpars.left
  r = ax.figure.subplotpars.right
  t = ax.figure.subplotpars.top
  b = ax.figure.subplotpars.bottom
  figw = float(w)/(r-l)
  figh = float(h)/(t-b)
  ax.figure.set_size_inches(figw, figh)

##########################################################

# Axis sizes
# In case of linear plotting you might need something like this: m = float(xmax-xmin)/(ymax-ymin)
#m = np.log(xmax-xmin)/np.log(ymax-ymin)
#mid_angle = np.arctan(m)/np.pi*180
xlogsize = float(np.log10(xmax/xmin))
ylogsize = float(np.log10(ymax/ymin))
m = xlogsize/ylogsize

print("Axis limits: 10^[(" + str(np.log10(xmax)) + " -> " + str(np.log10(xmin)) + ") x (" + str(np.log10(ymax)) + " ->" + str(np.log10(ymin)) + ")] = 10^[" + str(xlogsize) + " x " + str(ylogsize) + "]")
print("Plot logarithmic ratio: " + str(m) + "\n")
# START
max_roof  = gpu_roofs[0]["val"]
max_slope = mem_bottlenecks[0]["val"]

# Find maximum roof
for roof in gpu_roofs:
  if roof["val"] > max_roof:
    max_roof = roof["val"]

# Draw slopes
for slope in mem_bottlenecks:
  print("slope\t\"" + slope["name"] + "\"\t\t" + str(slope["val"]) + " GB/s")

  y = [0, max_roof]
  x = [float(yy)/slope["val"] for yy in y]
  ax.loglog(x, y, linewidth=1.0,
    linestyle='-.',
    marker="2",
    color="grey",
    zorder=10)

  # Label
  xpos = xmin*(10**(xlogsize*0.04))
  ypos = xpos*slope["val"]
  if ypos<ymin:
    ypos = ymin*(10**(ylogsize*0.02))
    xpos = ypos/slope["val"]
  pos = (xpos, ypos)

  # In case of linear plotting you might need something like this: trans_angle = np.arctan(slope["val"]*m)*180/np.pi
  #trans_angle = 45*m
  # print("\t" + str(trans_angle) + "°")

  ax.annotate(slope["name"] + ": " + str(slope["val"]) + " GB/s", pos,
    rotation=np.arctan(m/fig_ratio)*180/np.pi, rotation_mode='anchor',
    fontsize=11,
    ha="left", va='bottom',
    color="grey")

  # In the meantime: find maximum slope
  if slope["val"] > max_slope:
    max_slope = slope["val"]

print

# Draw roofs
for roof in gpu_roofs:
  print("roof\t\"" + roof["name"] + "\"\t\t" + str(roof["val"]) + " GFLOP/s")

  x = [roof["val"]/max_slope, xmax*10]
  ax.loglog(x, [roof["val"] for xx in x], linewidth=1.0,
    linestyle='-.',
    marker="2",
    color="grey",
    zorder=10)

  # Label
  ax.text(
    #roof["val"]/max_slope*10,roof["val"]*1.1,
    xmax/(10**(xlogsize*0.01)), roof["val"]*(10**(ylogsize*0.01)),
    roof["name"] + ": " + str(roof["val"]) + " GFLOP/s",
    ha="right",
    fontsize=11,
    color="grey")


#plt.xticks(list(plt.xticks()[0]) + [AI for n,AI in AI_v.items()], list(plt.xticks()[0]) + [str(AI) for n,AI in AI_v.items()])
for benchmark in AI_v:
  AI = AI_v[benchmark]
  print("benchmark\t\"" + benchmark + "\"\t\t" + str(AI) + " FLOP/Byte")

  plt.axvline(x=AI, dashes=[10, 10, 3, 10], linewidth=0.4, color="#AAAAAA")

  ax.text(
    AI/1.15, ymin*1.24,
    benchmark,
    fontsize=12,
    rotation=90,
    va="bottom",
    color="#888888")

# Draws datapoints
for point in datapoints:
  AI = point["AI"]
  if isinstance(AI,str):
    AI = AI_v[AI]

  ax.scatter(AI, point["GFLOP/s"], label=point["label"], zorder=100)

# Logarithmic axis labels format
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

# Set aspect
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.figlegend()
# plt.figlegend(loc="best")
plt.title("GEMM-Free LLM Roofline", fontsize=20)
plt.tight_layout()
set_size(fig_dimension*fig_ratio,fig_dimension)
plt.savefig("outputs/images/roofline.png")
plt.show()