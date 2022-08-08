import gc
import os
import copy
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn

from captum.attr import visualization as viz

from utils.captum.attribution_calls import *

def savefig(fig, folder, classname, explaintag, addti_info=""):
    if fig is not None:
        filepath = os.path.join(folder, classname+'_'+explaintag+addti_info+'.png')
        fig.savefig(filepath)
    else:
        gc.collect()
        torch.cuda.empty_cache()

def visualize_model(model, inputs, target, layerID=None, layer_name='', feature_mask=None, plt_fig_axis=None, show_plt=True, show_original=True,explain_out_img=None,classname=None):
    #layerID and layer_name are used in guidedGradCam
    #feature_mask used in featureAblation and shapleyValues

    fig, plt, _, tag = integratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=10, method='gausslegendre', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"10")

    fig, plt, _, tag = integratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=25, method='gausslegendre', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"25")

    fig, plt, _, tag = integratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=50, method='gausslegendre', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"50")

    fig, plt, _, tag = integratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=100, method='gausslegendre', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"100")

    fig, plt, _, tag = integratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=200, method='gausslegendre', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"200")

    fig, _, _, tag = noiseTunnelCONintegratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"10_10")

    fig, _, _, tag = noiseTunnelCONintegratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"25_10")

    fig, _, _, tag = noiseTunnelCONintegratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"10_2")

    fig, _, _, tag = noiseTunnelCONintegratedGradients(copy.deepcopy(model), inputs.clone(), target=target, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"25_2")

    #One of the differentiated Tensors does not require grad
    fig, _, _, tag = gradientShap(copy.deepcopy(model), inputs.clone(), n_samples=2, stdevs=0.0001, target=target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"2")

    fig, _, _, tag = gradientShap(copy.deepcopy(model), inputs.clone(), n_samples=10, stdevs=0.0001, target=target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag,"10")

    fig, _, _, tag = occlusion(copy.deepcopy(model), inputs.clone(), target, strides = (3, 4, 4), sliding_window_shapes=(3,50, 50), plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = saliency(copy.deepcopy(model), inputs.clone(), target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = deepLift(copy.deepcopy(model), inputs.clone(), target, useShap=True, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)
    
    fig, _, _, tag = deepLift(copy.deepcopy(model), inputs.clone(), target, useShap=False, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = inputXGradient(copy.deepcopy(model), inputs.clone(), target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = guidedBackprop(copy.deepcopy(model), inputs.clone(), target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    if layerID is not None:
        fig, _, _, tag = guidedGradCam(copy.deepcopy(model), layerID, inputs.clone(), target, device_ids=None, interpolate_mode='bilinear', attribute_to_layer_input=False, layer_name=layer_name, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
        savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = deconvolution(copy.deepcopy(model), inputs.clone(), target, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = featureAblation(copy.deepcopy(model), inputs.clone(), target, feature_mask=feature_mask, perturbations_per_eval=1, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

    fig, _, _, tag = shapleyValues(copy.deepcopy(model), inputs.clone(), target, feature_mask=feature_mask, perturbations_per_eval=1, useSampling=True, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)
    
    fig, _, _, tag = shapleyValues(copy.deepcopy(model), inputs.clone(), target, feature_mask=feature_mask, perturbations_per_eval=1, useSampling=False, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
    savefig(fig, explain_out_img, classname, tag)

def visualize_model_multilabel(model, inputs, target, class_names, layerID=None, layer_name='', feature_mask=None, plt_fig_axis=None, show_plt=True, show_original=True, explain_out_img=None):
    target = target.squeeze()
    pos_inds = np.argwhere(target==1)
    for i in pos_inds:
        classid = int(i[0])
        classname = class_names[classid]
        visualize_model(model, inputs, target=classid, layerID=layerID, layer_name=layer_name, feature_mask=feature_mask, plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original,explain_out_img=explain_out_img,classname=classname)