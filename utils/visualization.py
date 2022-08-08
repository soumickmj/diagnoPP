import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import os

epsilon = 0.00000000000000001
FOLD_NUM = range(3)

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

macro_metrics = ['Macro-Precision', 'Macro-Recall', 'Mean-Specificity', 'Macro-F1']
micro_metrics = ['Micro-Precision', 'Micro-Recall', 'Micro-F1']

root_path= '/mnt/memorial_shared/MEMoRIAL_SharedStorage_M1.2+4+7/Data/PublicDSs/Covid-19/IEEE8023_COVID_CHESTXRay/Finale/'
#ResNet34noMulLbl_intencrop_IEEE8023V2_60_40_00fold0_noCW_repeatGray_best.pth
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
    macro_metrics_fold_arr = np.zeros((classifier_num, len(macro_metrics), len(pathology_names), len(FOLD_NUM)))
    micro_metrics_fold_arr = np.zeros((classifier_num, len(micro_metrics), len(FOLD_NUM)))

    average_macro_metrics_arr = np.zeros((classifier_num, len(macro_metrics), len(pathology_names)))
    #sd_macro_metrics_arr = np.zeros((classifier_num, len(macro_metrics), len(pathology_names)))
    average_micro_metrics_arr = np.zeros((classifier_num, len(micro_metrics)))
    sd_micro_metrics_arr = np.zeros((classifier_num, len(micro_metrics)))

    for classifier in range(classifier_num):
        for fold in FOLD_NUM:
            classifier_path = classifier_paths[classifier].replace('fold0', 'fold'+str(fold)) if fold > 0 else classifier_paths[classifier]
            object = pd.read_pickle(root_path + classifier_path)
            mcm = object['mcm']
            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]
            #micro_metrics_fold_arr[classifier,0] = (tn + tp) / (tn+tp+fn+fp) # accuracy
            macro_metrics_fold_arr[classifier,0, :, fold] = tp / (tp + fp+epsilon) # precision
            macro_metrics_fold_arr[classifier,1, :, fold] = tp / (tp + fn + epsilon) #true positive rate or the sensitivity /recall
            macro_metrics_fold_arr[classifier,2, :, fold] = tn / (tn + fp + epsilon) #true negative rate / specificity
            macro_metrics_fold_arr[classifier,3, :, fold] = 2 * ((macro_metrics_fold_arr[classifier,0, :, fold] * macro_metrics_fold_arr[classifier,1, :, fold]) /
                                                                 (macro_metrics_fold_arr[classifier,0, :, fold] + macro_metrics_fold_arr[classifier,1, :, fold] + epsilon)) #F1
            sum_tp = np.sum(tp)
            sum_fp = np.sum(fp)
            sum_fn = np.sum(fn)
            micro_metrics_fold_arr[classifier,0, fold] = sum_tp / (sum_tp + sum_fp)  # M_precision
            micro_metrics_fold_arr[classifier,1, fold] = sum_tp / (sum_tp + sum_fn)  # M_recall
            micro_metrics_fold_arr[classifier,2, fold] = 2 * (micro_metrics_fold_arr[classifier,0, fold] * micro_metrics_fold_arr[classifier,1, fold]) / (
              micro_metrics_fold_arr[classifier,0, fold] + micro_metrics_fold_arr[classifier,1, fold])  # M_F1
        #macro_metrics_fold_arr[classifier, 3] = metrics.hamming_loss(object['y_true'].reshape((-1)), object['y_pred'].reshape((-1)))

    average_macro_metrics_arr = np.average(macro_metrics_fold_arr, axis=-1)
    #std_macro_metrics_arr = np.std(macro_metrics_fold_arr, axis=-1)
    average_micro_metrics_arr = np.average(micro_metrics_fold_arr, axis=-1)
    std_micro_metrics_arr = np.std(micro_metrics_fold_arr, axis=-1)
    print('F1 metric values for each classifier:')
    for idx, name in enumerate(classifier_names):
        print(name, ':', np.round(average_micro_metrics_arr[idx, 2],3), '+-', np.round(std_micro_metrics_arr[idx, 2],3))
    del micro_metrics_fold_arr, macro_metrics_fold_arr

    # class/pathology wise metrics
    for idx, pathology in enumerate(pathology_names):
        fig, ax = plt.subplots()
        fig= plt.figure(figsize=(10,10))
        x = np.arange(0.5, len(macro_metrics), 1)
        y = np.arange(0.5, len(classifier_names), 1)

        sn.heatmap(np.round(average_macro_metrics_arr[:,:,idx],3), cmap='Oranges', annot=True, fmt=".5g", annot_kws={"fontsize": 12, "fontweight":'bold'})

        plt.xticks(x, macro_metrics, fontsize=10, fontweight="bold")
        plt.yticks(y, classifier_names, rotation=50, fontsize=10, fontweight="bold")
        plt.savefig(save_path + '/' + pathology + '_metric.png', dpi=300, bbox_inches='tight')
        print(idx, pathology)
        plt.close(fig)

        # class/pathology level metrics

        fig, ax = plt.subplots()
        #fig = plt.figure(figsize=(10, 10))
        x = np.arange(0.5, len(micro_metrics), 1)
        y = np.arange(0.5, len(classifier_names), 1)

        sn.heatmap(np.round(average_micro_metrics_arr, 3), cmap='Oranges', annot=True, fmt=".5g")

        plt.xticks(x, micro_metrics, fontsize=10, fontweight="bold")
        plt.yticks(y, classifier_names, rotation=50, fontsize=10, fontweight="bold")
        plt.savefig(save_path + '/Classifier_level_metric.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

#plt.show()

if __name__ == '__main__':
    get_metric_values('/home/suhita/covid/images/metrics/')
    #get_confusion_matrices('/home/suhita/covid/images/confusion_matrix/')
