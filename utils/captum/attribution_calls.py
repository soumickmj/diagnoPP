import numpy as np
import gc

import torch
import torch.nn as nn

from captum.attr import IntegratedGradients, NoiseTunnel, GradientShap, Occlusion, Saliency, DeepLift, DeepLiftShap, InputXGradient
from captum.attr import GuidedBackprop, GuidedGradCam, Deconvolution, FeatureAblation, ShapleyValueSampling, ShapleyValues

from utils.captum.plotter import visualize

#Not implimented: Feature Permutation (https://captum.ai/api/feature_permutation.html)

def integratedGradients(model, inputs, baselines=None, target=None, additional_forward_args=None, n_steps=50, method='gausslegendre', return_convergence_delta=True, 
                        plt_fig_axis=None, show_plt=True, show_original=True):
    #baselines: define the starting point from which integral is computed. In the cases when baselines is not provided, internally use zero scalar corresponding to each input tensor.
    #additional_forward_args: If the forward function requires additional arguments other than the inputs for which attributions should not be computed
    #n_steps: The number of steps used by the approximation method. (captum default:50)
    #method: Method for approximating the integral, one of riemann_right, riemann_left, riemann_middle, riemann_trapezoid or gausslegendre.
    #plt_fig_axis: (4viz) Tuple of matplotlib.pyplot.figure and axis on which to visualize. If None is provided, then a new figure and axis are created. 
    try:
        inputs = inputs.contiguous()
        ig = IntegratedGradients(model)
        attributions, delta = ig.attribute(inputs, baselines=baselines, target=target, n_steps=n_steps, method=method, return_convergence_delta=return_convergence_delta)    

        fig, ax = visualize(attributions, inputs,
                            outlier_perc=1,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Integrated Gradients',
                            show_original=show_original)
        del model, inputs, ig  
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, (attributions, delta), "integratedGradients"
    except Exception as e:
        print("Couldn't execute: integratedGradients"+str(e))
        return None, None, None, None

def noiseTunnel(attrib_method, inputs, target=None, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', title=None, plt_fig_axis=None, show_plt=True, show_original=True):
    #TODO: Use **kwargs for n_steps, as only needed for integrated gradients
    try:
        inputs = inputs.contiguous()
        nt = NoiseTunnel(attrib_method)
        attributions = nt.attribute(inputs, n_samples=n_samples, nt_type=nt_type, target=target, n_steps=n_steps)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title=title,
                            show_original=show_original)
        del attrib_method, inputs, nt  
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions
    except Exception as e:
        print("Couldn't execute: noiseTunnel"+str(e))
        return None, None, None

def noiseTunnelCONintegratedGradients(model, inputs, target=None, n_steps=10, n_samples=10, nt_type='smoothgrad_sq', plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        ig = IntegratedGradients(model)
        fig, ax, attributions = noiseTunnel(ig, inputs, target=target, n_steps=n_steps, n_samples=n_samples, nt_type=nt_type, 
                    title='Noise Tunnel for Integrated Gadients', plt_fig_axis=plt_fig_axis, show_plt=show_plt, show_original=show_original)
        return fig, ax, attributions, "noiseTunnelCONintegratedGradients"
    except Exception as e:
        print("Couldn't execute: noiseTunnelCONintegratedGradients"+str(e))
        return None, None, None, None

def gradientShap(model, inputs, n_samples=50, stdevs=0.0001, target=None, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        gs = GradientShap(model)

        # Defining baseline distribution of images
        rand_img_dist = torch.cat([inputs * 0, inputs * 1])
        rand_img_dist.requires_grad=True
        rand_img_dist = rand_img_dist.contiguous()

        attributions = gs.attribute(inputs,
                                    n_samples=n_samples,
                                    stdevs=stdevs,
                                    baselines=rand_img_dist,
                                    target=target)

        fig, ax = visualize(attributions, inputs,
                            sign = "absolute_value",
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='GradientShap',
                            show_original=show_original)
        del model, inputs, gs, rand_img_dist
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "gradientShap"
    except Exception as e:
        print("Couldn't execute: gradientShap"+str(e))
        return None, None, None, None

def occlusion(model, inputs, target, baselines=0, strides = (3, 8, 8), sliding_window_shapes=(3,15, 15), plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        oc = Occlusion(model)
        attributions = oc.attribute(inputs,
                                        strides = strides,
                                        target=target,
                                        sliding_window_shapes=sliding_window_shapes,
                                        baselines=0)

        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Occlusion',
                            show_original=show_original)
        del model, inputs, oc
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "occlusion"
    except Exception as e:
        print("Couldn't execute: occlusion"+str(e))
        return None, None, None, None

def saliency(model, inputs, target, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        sln = Saliency(model)
        grads = sln.attribute(inputs, target=target)
        fig, ax = visualize(grads, inputs,
                            sign="absolute_value",
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Saliency: Overlayed Gradient Magnitudes',
                            show_original=show_original)
        del model, inputs, sln
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, grads, "saliency"
    except Exception as e:
        print("Couldn't execute: saliency"+str(e))
        return None, None, None, None

def deepLift(model, inputs, target, useShap=True, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        if useShap:
            dl = DeepLiftShap(model)
            title = "Overlayed DeepLiftShap"
            tag = "deepLiftShap"
            in_sz=inputs.size()
            baselines = torch.randn([in_sz[0]*10, in_sz[1], in_sz[2], in_sz[3]], device=inputs.device, requires_grad=True) * 0.001
        else:
            dl = DeepLift(model)
            title = "Overlayed DeepLift"
            tag = "deepLift"
            baselines = None
        attributions = dl.attribute(inputs, baselines=baselines, target=target)
        fig, ax = visualize(attributions, inputs,
                            sign="all",
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title=title,
                            show_original=show_original)
        del model, inputs, dl
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, tag
    except Exception as e:
        print("Couldn't execute: deepLift"+("deepLiftShap" if useShap else "deepLift")+": "+str(e))
        return None, None, None, None

def inputXGradient(model, inputs, target, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        ixg = InputXGradient(model)
        attributions  = ixg.attribute(inputs, target=target)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Input X Gradient',
                            show_original=show_original)
        del model, inputs, ixg
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "inputXGradient"
    except Exception as e:
        print("Couldn't execute: inputXGradient"+str(e))
        return None, None, None, None

def guidedBackprop(model, inputs, target, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        gbp = GuidedBackprop(model)
        attributions  = gbp.attribute(inputs, target=target)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Guided Backprop',
                            show_original=show_original)
        del model, inputs, gbp
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "guidedBackprop"
    except Exception as e:
        print("Couldn't execute: guidedBackprop"+str(e))
        return None, None, None, None

def guidedGradCam(model, layer, inputs, target, device_ids=None, interpolate_mode='bilinear', attribute_to_layer_input=False, layer_name='', plt_fig_axis=None, show_plt=True, show_original=True):
    #layer: Layer for which GradCAM attributions are computed. Currently, only layers with a single tensor output are supported.
    #device_ids: Device ID list, necessary only if forward_func applies a DataParallel model. This allows reconstruction of intermediate outputs from batched results across devices. If forward_func is given as the DataParallel model itself, then it is not necessary to provide this argument.
    #interpolate_mode: Method for interpolation, which must be a valid input interpolation mode for torch.nn.functional.
    #attribute_to_layer_input:Indicates whether to compute the attribution with respect to the layer input or output in LayerGradCam. If attribute_to_layer_input is set to True then the attributions will be computed with respect to layer inputs, otherwise it will be computed with respect to layer outputs. Note that currently it is assumed that either the input or the output of internal layer, depending on whether we attribute to the input or output, is a single tensor.
    try:
        inputs = inputs.contiguous()
        ggc = GuidedGradCam(model, layer, device_ids)
        attributions  = ggc.attribute(inputs, target=target, interpolate_mode=interpolate_mode, attribute_to_layer_input=attribute_to_layer_input)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Guided GradCam: '+layer_name,
                            show_original=show_original)
        del model, inputs, ggc
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "guidedGradCam_"+layer_name
    except Exception as e:
        print("Couldn't execute: guidedGradCam_"+layer_name+": "+str(e))
        return None, None, None, None

def deconvolution(model, inputs, target, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        dc = Deconvolution(model)
        attributions  = dc.attribute(inputs, target=target)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Deconvolution',
                            show_original=show_original)
        del model, inputs, dc
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "deconvolution"
    except Exception as e:
        print("Couldn't execute: deconvolution"+str(e))
        return None, None, None, None

def featureAblation(model, inputs, target, baselines=None, feature_mask=None, perturbations_per_eval=1, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        ablator = FeatureAblation(model)
        attributions  = ablator.attribute(inputs, baselines=baselines, target=target, feature_mask=feature_mask, perturbations_per_eval=perturbations_per_eval)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title='Feature Ablation',
                            show_original=True)
        del model, inputs, ablator
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, "featureAblation"
    except Exception as e:
        print("Couldn't execute: featureAblation: "+str(e))
        return None, None, None, None
    
def shapleyValues(model, inputs, target, baselines=None, feature_mask=None, perturbations_per_eval=1, useSampling=True, plt_fig_axis=None, show_plt=True, show_original=True):
    try:
        inputs = inputs.contiguous()
        if useSampling:
            sv = ShapleyValueSampling(model)
            title = 'Shapley Value Sampling'
            tag = "shaplingValueSampling"
        else:
            sv = ShapleyValues(model)
            title = 'Shapley Values'
            tag="shapleyValues"
        attributions  = sv.attribute(inputs, baselines=baselines, target=target, feature_mask=feature_mask, perturbations_per_eval=perturbations_per_eval)
        fig, ax = visualize(attributions, inputs,
                            plt_fig_axis=plt_fig_axis,
                            show_plt=show_plt, 
                            title=title,
                            show_original=show_original)
        del model, inputs, sv
        gc.collect()
        torch.cuda.empty_cache()
        return fig, ax, attributions, tag
    except Exception as e:
        print("Couldn't execute: shapleyValues"+("shaplingValueSampling" if useSampling else "shapleyValues")+": "+str(e))
        return None, None, None, None