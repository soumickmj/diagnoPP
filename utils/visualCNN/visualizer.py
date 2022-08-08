import utils.visualCNN as viz
def visualize_model(model, layerID, filter_position, im_path, classID, inputs, file_name_to_export):
    if isinstance(model.net[0].net, models.resnet.ResNet):
        model_features = model.net[0].net.layer4
    elif isinstance(model.net[0].net, models.inception.Inception3):
        model_features = model.net[0].net.Mixed_7c
    else:
        model_features = model.net[0].net.features
    viz.cnn_layer_visualization.execute(model, layerID, filter_position)
    viz.deep_dream.execute(model, layerID, filter_position, im_path)
    viz.generate_class_specific_samples.execute(model, classID)
    viz.generate_regularized_class_specific_samples.execute(model, classID)
    viz.grad_times_image.execute(model, inputs, classID, os.path.join(file_name_to_export, 'grad_times_image'))
    viz.gradcam.execute(model, layerID, inputs, classID, file_name_to_export)
    viz.guided_backprop.execute(model, inputs, classID, file_name_to_export)
    viz.guided_gradcam.execute(model, layerID, inputs, classID, file_name_to_export)
    viz.integrated_gradients.execute(model, inputs, classID, file_name_to_export, steps=100)
    viz.inverted_representation.execute(model, layerID, inputs, image_size=512)
    viz.layer_activation_with_guided_backprop.execute(model, layerID, filter_position, inputs, classID, file_name_to_export)
    viz.scorecam.execute(model, layerID, inputs, classID, file_name_to_export)
    viz.smooth_grad.execute(model, inputs, classID, file_name_to_export, param_n = 50,param_sigma_multiplier = 4)
    viz.vanilla_backprop.execute(model, inputs, classID, file_name_to_export)

