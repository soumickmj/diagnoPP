import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn

from captum.attr import visualization as viz

# default_cmap = LinearSegmentedColormap.from_list('custom blue', 
#                                                  [(0, '#ffffff'),
#                                                   (0.25, '#000000'),
#                                                   (1, '#000000')], N=256)

# default_cmap = None

default_cmap = "magma"

def visualize_single(attributions, inputs, method='blended_heat_map', sign='positive', outlier_perc=2, show_colorbar=True, plt_fig_axis=None, title=None, show_plt=True):
    fig, ax = viz.visualize_image_attr(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)), #attributions
                                        np.transpose(inputs.squeeze().cpu().detach().numpy(), (1,2,0)), #original image
                                        method=method, #method of visualization: heat_map(default), blended_heat_map, original_image, masked_image, alpha_scaling 
                                        cmap=default_cmap,
                                        show_colorbar=show_colorbar, #Displays colorbar for heatmap below the visualization.
                                        sign=sign, #Chosen sign of attributions to visualize: positive (default), absolute_value, negative, all
                                        outlier_perc=outlier_perc, #Top attribution values which correspond to a total of outlier_perc percentage of the total attribution are set to 1 and scaling is performed using the minimum of these values. For sign=`all`, outliers a nd scale value are computed using absolute value of attributions. Default: 2
                                        plt_fig_axis=plt_fig_axis, #Tuple of matplotlib.pyplot.figure and axis on which to visualize. If None is provided, then a new figure and axis are created.
                                        use_pyplot=show_plt, #If true (default), uses pyplot to create and show figure and displays the figure after creating. If False, uses Matplotlib object oriented API and simply returns a figure object without showing.
                                        title=title)
    return fig, ax

def visualize_multi(attributions, inputs, methods=["original_image", "heat_map"], signs = ["all", "positive"], outlier_perc=2, titles=None, show_colorbar=True, plt_fig_axis=None, show_plt=True):
    fig, ax = viz.visualize_image_attr_multiple(np.transpose(attributions.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                np.transpose(inputs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                                methods = methods,
                                                signs = signs,
                                                outlier_perc=outlier_perc,
                                                cmap=default_cmap,
                                                show_colorbar=show_colorbar, 
                                                titles=titles,
                                                use_pyplot=show_plt,
                                                alpha_overlay=0.5)#,
                                                # plt_fig_axis=plt_fig_axis)
    return fig, ax

def visualize(attributions, inputs, method='blended_heat_map', sign='positive', show_original=True, outlier_perc=2, show_colorbar=True, plt_fig_axis=None, title=None, show_plt=True):
    if show_original:
        fig, ax = visualize_multi(attributions, inputs,
                                    methods=["original_image", method], signs = ["all", sign], outlier_perc=outlier_perc, 
                                    titles=['Original Image', title], show_colorbar=show_colorbar, plt_fig_axis=plt_fig_axis, show_plt=show_plt)
    else:
        fig, ax = visualize_single(attributions, inputs, 
                                    method=method, sign=sign, outlier_perc=outlier_perc, 
                                    show_colorbar=show_colorbar, plt_fig_axis=plt_fig_axis, title=title, show_plt=show_plt)

    return fig, ax