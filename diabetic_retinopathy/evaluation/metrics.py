import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt


class ConfusionMatrix(object):
    def __init__(self, y_pred, y_true):
        # initialize parameters
        # self.num_class = num_class
        self.y_pred = y_pred
        self.y_true = y_true

        cm = self.confusion_matrix_show(self.y_true, self.y_pred)
        accuracy, precision, recall, f1_score, sensitivity, specificity = self.accuracy(cm)
        self.confusion_matrix_plot(cm)
        self.roc_curve_plot(y_true, y_pred)
        stats_text = "\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}\nSensitivity={:0.3f}\nSpecificity={:0.3f}"\
            .format(accuracy, precision, recall, f1_score, sensitivity, specificity)
        print(stats_text)

    def confusion_matrix_show(self, y_true, y_pred):
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print(cm)
        return cm

    def accuracy(self, cm):
        accuracy = np.trace(cm) / float(np.sum(cm))
        precision = cm[1, 1] / sum(cm[1, :])
        recall = cm[1, 1] / sum(cm[1, :])
        f1_score = 2 * precision * recall / (precision + recall)
        sensitivity = cm[1, 1] / sum(cm[:, 1])
        specificity = cm[0, 0] / sum(cm[:, 0])
        return accuracy, precision, recall, f1_score, sensitivity, specificity

    def confusion_matrix_plot(self, cm):
        plt.figure(dpi=120)
        group_names = ['True_negative', 'False_negative', 'False_negative', 'True_positive']
        group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
        labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2, 2)
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
        plt.xlabel('Predict')
        plt.ylabel('True')
        plt.title('Confusion matrix')
        plt.show()

    def roc_curve_plot(self, y_true, y_pred):
        auc = roc_auc_score(y_true, y_pred)
        # auc = roc_auc_score(y_test,clf.decision_function(X_test))
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()