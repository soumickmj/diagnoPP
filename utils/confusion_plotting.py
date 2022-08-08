import io
import PIL.Image
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from pylab import *

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

def plot_confusion_matrix(y_true, y_pred, class_ids, class_names=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if class_names is None:
        class_names = class_ids    

    # Compute confusion matrix
    cm = multilabel_confusion_matrix(y_true, y_pred) ##########
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    subplots_adjust(hspace=0.000)
    number_of_subplots=cm.shape[0]
    from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
    for i,v in enumerate(range(number_of_subplots)):
        v = v+1
        ax = subplot(number_of_subplots,1,v)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm[i])
        disp.plot(ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig, ax = plt.subplots()
    #fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax

def cm_as_img_pyt(y_true, y_pred, class_ids, class_names=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
                          
    from torchvision.transforms import ToTensor
    cm, _ = plot_confusion_matrix(y_true, y_pred, class_ids, class_names, normalize, title, cmap)
    buf = io.BytesIO()
    cm.savefig(buf, format='png')
    buf.seek(0)
    cm_img = ToTensor()(PIL.Image.open(buf))

    return cm_img

def cm_as_img_tf(y_true, y_pred, class_ids, class_names=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    import tensorflow as tf
    cm, _ = plot_confusion_matrix(y_true, y_pred, class_ids, class_names, normalize, title, cmap)
    buf = io.BytesIO()
    cm.savefig(buf, format='png')
    buf.seek(0)
    cm_img = tf.image.decode_png(buf.getvalue(), channels=4) 
    # cm_img = tf.expand_dims(cm_img, 0) #add batch dim

    return cm_img