import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import os

epsilon = 0.00000000000000001
pathology_names = np.sort(["Pneumonia", "Viral Pneumonia", "Bacterial Pneumonia", "Fungal Pneumonia",
                    "No Finding", "COVID-19", "SARS", "MERS", "ARDS", "Streptococcus", "Pneumocystis", "Klebsiella", "Chlamydophila", "Legionella"])

ml_classifier_names = [
    'DenseNet161',
    #'DenseNet161_v1',
    'IncepResV2',
    #'IncepResV2_v1',
    'InceptionV3',
    #'InceptionV3_v1',
    #'InceptionV4',
    'ResNet18',
    #'ResNet18_v1',
    'ResNet34',
    #'ResNet34_v1',
    # 'ResNet152',
    # 'ResNext50',
    # 'ResNext101',
    # 'xVGG19_BN',
    # 'xWide_Resnet50',
    'Ensemble'
    #'Ensemble_Weighted'
 ]

mc_classifier_names = [
    #'MC_DenseNet161',
    #'MC_DenseNet161_v1',
    #'MC_IncepResV2',
    #'MC_IncepResV2_v1',
    #'MC_InceptionV3',
    #'MC_InceptionV3_v1',
   # 'MC_InceptionV4',
    #'MC_ResNet18',
    #'MC_ResNet18_v1',
    #'MC_ResNet34',
    #'MC_ResNet34_v1',
 ]

class_wise_metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1']
class_level_metrics = ['M-Precision', 'M-Recall', 'M-F1']

root_path= r"/mnt/memorial_shared/MEMoRIAL_SharedStorage_M1.2+4+7/Data/PublicDSs/Covid-19/IEEE8023_COVID_CHESTXRay/Finale/"
ml_classifier_pickle_paths = [
    'DenseNet161_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
   # 'DenseNet161_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    'IncepResV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    #'IncepResV2_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    'InceptionV3_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    #'InceptionV3_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    #'InceptionV4_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    'ResNet18_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    #'ResNet18_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    'ResNet34_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    #'ResNet34_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNet152_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNext50_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNext101_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'xVGG19_BN_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray79.pkl',
    # 'xWide_Resnet50_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    'EnsambleV1_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray_test.pkl',
    #'EnsambleV2_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray_test.pkl'
 ]

multiclass_classifier_pickle_paths = [
    # 'DenseNet161noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'DenseNet161noMulLbl_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'IncepResV2noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'IncepResV2noMulLbl_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'InceptionV3noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'InceptionV3noMulLbl_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNet18noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNet18noMulLbl_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNet34noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray249.pkl',
    # 'ResNet34noMulLbl_intencrop_IEEE8023_60_40_00fold0_noCW_repeatGray249.pkl'
 ]


def get_confusion_matrices(save_path):
    classifier_paths = ml_classifier_pickle_paths + multiclass_classifier_pickle_paths
    classifier_names = ml_classifier_names + mc_classifier_names
    for ctr, path in enumerate(classifier_paths):
        object = pd.read_pickle(root_path+path)
        mcm = object['mcm']

        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
        #print(tp.shape)
        #accuracy = (tn + tp) / (tn+tp+fn+fp)
        precision = tp / (tp + fp+epsilon)
        recall = tp / (tp + fn + epsilon) #true positive rate or the sensitivity
        specificity = tn / (tn + fp + epsilon) #true negative rate
        #fallout = fp / (fp + tn + epsilon) #false positive rate
        #missrate = fn / (fn + tp + epsilon) #false negative rate
        f1 = 2 * ((precision*recall) / (precision+recall+epsilon))


        fig, ax = plt.subplots(4,4, figsize=(20, 10))
        row_ctr = 0
        col_ctr = 0
        for idx in range(14):
            if (idx >0) and (idx % 4 == 0):
                row_ctr += 1
                col_ctr = 0
            cls = np.array([[tn[idx], fp[idx]], [fn[idx], tp[idx]]])
            ax[row_ctr, col_ctr] = sn.heatmap(cls, cmap='Oranges', annot=True, fmt=".5g", cbar=False, ax=ax[row_ctr,col_ctr])
            ax[row_ctr, col_ctr].title.set_text(pathology_names[idx])
            col_ctr += 1

        plt.savefig(os.path.join(save_path, classifier_names[ctr] + '.png'))
        print(classifier_names[ctr])
        plt.close(fig)


# metrics for classes




def get_metric_values(save_path):

    classifier_paths = ml_classifier_pickle_paths + multiclass_classifier_pickle_paths
    classifier_names = ml_classifier_names + mc_classifier_names

    classifier_num = len(classifier_paths)
    classifier_metrics = np.zeros((classifier_num, len(class_wise_metrics), len(pathology_names)))
    combined_class_metrics = np.zeros((classifier_num, len(class_level_metrics)))

    hamming_loss = 0.
    for classifier in range(classifier_num):
        object = pd.read_pickle(root_path + classifier_paths[classifier])
        mcm = object['mcm']
        tn = mcm[:, 0, 0]
        tp = mcm[:, 1, 1]
        fn = mcm[:, 1, 0]
        fp = mcm[:, 0, 1]
        classifier_metrics[classifier,0] = (tn + tp) / (tn+tp+fn+fp) # accuracy
        classifier_metrics[classifier,1] = tp / (tp + fp+epsilon) # precision
        classifier_metrics[classifier,2] = tp / (tp + fn + epsilon) #true positive rate or the sensitivity /recall
        classifier_metrics[classifier,3] = tn / (tn + fp + epsilon) #true negative rate / specificity
        classifier_metrics[classifier,4] = 2 * ((classifier_metrics[classifier,1] * classifier_metrics[classifier,2]) / (classifier_metrics[classifier,1] + classifier_metrics[classifier,2] + epsilon)) #F1
        sum_tp = np.sum(tp)
        sum_fp = np.sum(fp)
        sum_fn = np.sum(fn)
        combined_class_metrics[classifier,0] = sum_tp / (sum_tp + sum_fp)  # M_precision
        combined_class_metrics[classifier,1] = sum_tp / (sum_tp + sum_fn)  # M_recall
        combined_class_metrics[classifier,2] = 2 * (combined_class_metrics[classifier,0] * combined_class_metrics[classifier,1]) / (
          combined_class_metrics[classifier,0] + combined_class_metrics[classifier,1])  # M_F1
        #combined_class_metrics[classifier, 3] = metrics.hamming_loss(object['y_true'].reshape((-1)), object['y_pred'].reshape((-1)))


    # class/pathology wise metrics
    for idx, pathology in enumerate(pathology_names):
        fig, ax = plt.subplots()
        # fig= plt.figure(figsize=(10,10))
        x = np.arange(0.5,len(class_wise_metrics),1)
        y= np.arange(0.5,len(classifier_names),1)


        sn.heatmap(np.round(classifier_metrics[:,:,idx],3), cmap='Oranges', annot=True, fmt=".5g", annot_kws={"fontsize": 12, "fontweight":'bold'})

        plt.xticks(x, class_wise_metrics, fontsize=10, fontweight="bold")
        plt.yticks(y, classifier_names, rotation=50, fontsize=10, fontweight="bold")
        # plt.ylabel('Classifiers', fontsize=12)
        # plt.xlabel('Metrics', fontsize=12)
        plt.savefig(save_path +'/'+ pathology + '_metric.png', dpi=300, bbox_inches='tight')
        print(idx, pathology)
        plt.close(fig)

        # class/pathology level metrics

        fig, ax = plt.subplots()
        # fig = plt.figure(figsize=(10, 10))
        x = np.arange(0.5,len(class_level_metrics),1)
        y = np.arange(0.5,len(classifier_names),1)

        sn.heatmap(np.round(combined_class_metrics, 3), cmap='Oranges', annot=True, fmt=".5g", annot_kws={"fontsize": 12, "fontweight":'bold'})

        plt.xticks(x, class_level_metrics, fontsize=10, fontweight="bold")
        plt.yticks(y, classifier_names, rotation=50, fontsize=10, fontweight="bold")
        # plt.ylabel('Classifiers', fontsize=15)
        # plt.xlabel('Metrics', fontsize=15)
        plt.savefig(save_path + '/Classifier_level_metric.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

#plt.show()

if __name__ == '__main__':
    get_metric_values('/home/suhita/PycharmProjects/covid-19-segment-n-explain/resources/images/')
    # get_confusion_matrices('resources/images/confusion_matrix')
