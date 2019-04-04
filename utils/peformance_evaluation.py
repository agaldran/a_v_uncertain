import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve, matthews_corrcoef

def save_log(roc, thr_stats, pth='../results/roc_stats.txt'):
    print('Results saved to '+pth)
    with open(pth, 'w') as fl:
        fl.write('Performance Analysis: \n' )
        fl.write("AUC ROC: %0.4f \n" % roc['auc'])
        fl.write("Optimal Threshold 4 Youden (OTY): %0.4f \n" % roc['optimal_thr_youden'])
        fl.write("Optimal Threshold 4 Accuracy (OTA): %0.4f \n" % roc['optimal_thr_acc'])
        fl.write("Accuracy at OTA: %0.4f \n" % thr_stats['acc'])
        fl.write("Balanced Accuracy at OT: %0.4f \n" % thr_stats['balanced_acc'])
        fl.write("Sensitivity at OT: %0.4f \n" % thr_stats['se'])
        fl.write("Specificity at OT: %0.4f \n" % thr_stats['sp'])
        fl.write("F1-score at OT: %0.4f \n" % thr_stats['fscore'])
        fl.write("MCC-score at OT: %0.4f \n" % thr_stats['mcc'])


def perf_stats(labels, predicted_classes):
    """
    INPUT
    labels = true class labels
    preds = predicted class labels

    OUTPUT
    stats is a dictionary
    stats.confusionMat
               Predicted Classes
                    p'    n'
              ___|_____|_____|
       Actual  p |     |     |
      Classes  n |     |     |

    stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
    stats.precision = TP / (TP + FP)                  % for each class label
    stats.sensitivity = TP / (TP + FN)                % for each class label
    stats.specificity = TN / (FP + TN)                % for each class label
    stats.recall = sensitivity                        % for each class label
    stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label

    TP: true positive, TN: true negative,
    FP: false positive, FN: false negative

    """

    stats = {}
    alpha = 1e-8
    cmat = confusion_matrix(labels, predicted_classes)
    tn, fp, fn, tp = cmat.ravel()
    stats['conf_mat'] = cmat
    stats['acc'] = np.float(tp+tn)/(np.float(tp+fp+fn+tn) + alpha)
    stats['se'] = np.float(tp)/(np.float(tp+fn) + alpha)
    stats['sp'] = np.float(tn) / (np.float(fp + tn) + alpha)
    stats['precision'] = np.float(tp) / (np.float(tp+fp) + alpha)
    stats['recall'] = stats['se']
    stats['fscore'] = np.float(2*tp) / (np.float(2*tp+fp+fn) + alpha)
    stats['balanced_acc'] = 0.5*(stats['se']+stats['sp'])
    stats['mcc'] = matthews_corrcoef(labels, predicted_classes)

    return stats

def roc_analysis(labels, scores):
    roc = {}
    fpr, tpr, ts = roc_curve(labels, scores, pos_label=1)
    # Compute Youden-maximizing threshold
    J = tpr + (1 - fpr) - 1  # Youden Index
    jmax = np.where(J == np.max(J))
    opt_thr_youden = ts[jmax]
    if len(opt_thr_youden) > 1:
        opt_thr_youden = opt_thr_youden[0]

    # Compute accuracy-maximizing threshold
    max_acc = -1
    opt_thresh_acc = -1
    for thresh in ts:
        scores_thr = scores > thresh
        acc = np.sum(labels == scores_thr) / len(labels)
        if acc > max_acc:
            opt_thresh_acc = thresh
            max_acc = acc

    roc['fpr'] = fpr
    roc['tpr'] = tpr
    roc['auc'] = roc_auc_score(labels, scores)
    roc['thresholds'] = ts[::-1]

    roc['optimal_thr_youden'] = opt_thr_youden
    roc['jmax'] = jmax

    roc['optimal_thr_acc'] = opt_thresh_acc
    roc['accuracy_acc_max'] = max_acc

    return roc
