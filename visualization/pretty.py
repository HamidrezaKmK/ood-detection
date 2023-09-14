"""
This file contains all the functions that visualize plots in a color-themed and pretty way!
"""

import numpy as np
import typing as th
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter
from math import log10
from enum import Enum, auto

class ColorTheme(Enum):
    OOD = "#A02C30" 
    OOD_SECONDARY = "#DF8A8A" 
    IN_DISTR = "#1F78B4" 
    IN_DISTR_SECONDARY = "#A6CBE3" 
    GENERATED = "#dfc27d" 
    DENSITY = "#dfc27d"   


hashlines = ['////', '\\\\\\\\', '|||', '---', '+', 'x', 'o', '0', '.', '*']
line_styles = ['-', '--', '-.', ':']

def plot_histogram(
    x_values: np.array, 
    labels: th.List[str],
    colors: th.List[str],
    x_label: th.Optional[str]=None,
    scale: int = 0,
    figsize: tuple = (10, 6),
    bins: int = 50,
    binwidth: th.Optional[float] = None,
    legend_loc: th.Optional[str] = None,
    xlim: th.Optional[tuple] = None,
):
    """
    Plots overlapping histograms.
    
    Parameters:
        x_values (list of numpy arrays): Data values for histograms.
        labels (list of str): Labels for each histogram.
        colors (list of str): Colors for each histogram.
    """
    
    # Ensure input lists are of the same length
    assert len(x_values) == len(labels) == len(colors), "Input lists must have the same length"
    
    # Set seaborn style with custom grid
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.01})
    
    # Adjust hatch density
    plt.rcParams['hatch.linewidth'] = 0.5 # previous value was 1
    # plt.rcParams['hatch.density'] = 5 
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for x, label, color in zip(x_values, labels, colors):
        if binwidth is not None:
            bin_args = {'binwidth': binwidth}
        else:
            bin_args = {'bins': bins}
        # Plotting the histogram (density ensures it's normalized)
        sns.histplot(x, kde=True, **bin_args,
                     ax=ax, 
                     color=color, label=label, element="step",
                     stat="density", common_norm=False, 
                    #  kde_kws=,
                     hatch=hashlines[idx%len(hashlines)])
        
        # sns.histplot(x, bins='auto', color=color, kde=False, label=label, stat='density', alpha=0.5, linewidth=0)
        
        # Plotting the KDE on top
        # sns.kdeplot(x, color=color, linestyle=line_styles[idx % len(line_styles)])
        idx += 1
        
    # Adjust y-axis label based on the scale
    if scale != 0:
        ax.yaxis.set_major_formatter(lambda x, _: f'{x * 10 ** (scale):.1f}')
        ax.set_ylabel(f'Density $\\times 10^{{-{scale}}}$')
    else:
        ax.set_ylabel('Density')
    
    if legend_loc:
        # Add legend to the left of the plot
        ax.legend(loc=legend_loc)
    
    if x_label is not None:
        ax.set_xlabel(x_label)
    if xlim is not None:
        ax.set_xlim(xlim)
        
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_kde(
    x_values: np.array, 
    labels: th.List[str],
    colors: th.List[str],
    x_label: th.Optional[str]=None,
    scale: int = 0,
    figsize: tuple = (10, 6),
    bins: int = 50,
    binwidth: th.Optional[float] = None,
    legend_loc: th.Optional[str] = None,
    xlim: th.Optional[tuple] = None,
):
    """
    Plots KDE for given data.

    Parameters:
    - x_values: List of numpy arrays
    - labels: List of labels corresponding to x_values
    - colors: List of colors corresponding to x_values
    - scale: Integer value to scale the ylabel
    """
    
    # Ensure input lists are of the same length
    assert len(x_values) == len(labels) == len(colors), "Input lists must have the same length"
    
    # Set seaborn style with custom grid
    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.2})
    
    # Adjust hatch density
    plt.rcParams['hatch.linewidth'] = 0.5 # previous value was 1
    # plt.rcParams['hatch.density'] = 5 
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    idx = 0
    for x, label, color in zip(x_values, labels, colors):
        density = sns.kdeplot(x, bw_adjust=0.5, color=color).get_lines()[-1].get_data()
        ax.fill_between(density[0], 0, density[1], color=color, label=label, alpha=0.5, hatch=hashlines[idx % len(hashlines)])
        idx += 1
        
    # Adjust y-axis label based on the scale
    if scale != 0:
        ax.yaxis.set_major_formatter(lambda x, _: f'{x * 10 ** (scale):.1f}')
        ax.set_ylabel(f'Density $\\times 10^{{-{scale}}}$')
    else:
        ax.set_ylabel('Density')
    
    if legend_loc:
        # Add legend to the left of the plot
        ax.legend(loc=legend_loc)
    
    if x_label is not None:
        ax.set_xlabel(x_label)
    if xlim is not None:
        ax.set_xlim(xlim)
        
    plt.tight_layout()
    plt.show()

