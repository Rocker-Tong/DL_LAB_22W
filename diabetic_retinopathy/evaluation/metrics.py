import tensorflow as tf
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # initialize parameters
        # self.num_class = num_class
        self.confusion_matrix = self.add_weight(name="confusion_matrix",
                                                shape=(2, 2),
                                                initializer="zeros")

    def update_state(self, labels, predictions, sample_weight=None):
        # Update parameters
        labels = tf.cast(labels, dtype=tf.int32)
        # labels = tf.argmax(labels, axis=-1)
        predictions = tf.argmax(predictions, axis=-1)
        self.confusion_matrix = tf.math.confusion_matrix(tf.squeeze(labels),
                                                         tf.squeeze(predictions),
                                                         num_classes=2,
                                                         dtype=tf.int32)


        # one_hot_true = tf.one_hot(labels, depth=self.2)
        # one_hot_pred = tf.one_hot(predictions, depth=self.2)

        # self.confusion_matrix.assign_add(
        #     tf.matmul(one_hot_true, one_hot_pred, transpose_a=True))

    def result(self):
        return self.confusion_matrix

    # def reset_states(self):
    #     # Reset the state of the metric
    #     self.confusion_matrix.assign(
    #         tf.zeros([2, 2]))


def confusion_matrix_show(y_true, y_pred):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(cm)
    return cm


def accuracy(cm):
    accuracy = np.trace(cm) / float(np.sum(cm))
    precision = cm[1, 1] / sum(cm[1, :])
    recall = cm[1, 1] / sum(cm[1, :])
    f1_score = 2 * precision * recall / (precision + recall)
    sensitivity = cm[1, 1] / sum(cm[:, 1])
    specificity = cm[0, 0] / sum(cm[:, 0])
    return accuracy, precision, recall, f1_score, sensitivity, specificity


def confusion_matrix_plot(cm):
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
    return


def roc_curve_plot(y_true, y_pred):
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
    return

