import numpy as np
import matplotlib.pyplot as plt


def generate_ROC(classes, decision_stats, roc_type=1, plot_title='ROC Curve', label='Label', verbose=False, show=True):
    decision_stats, classes = (np.array(list(t)) for t in zip(*sorted(zip(decision_stats, classes))))

    num_h1 = classes.sum()
    num_h0 = len(classes) - num_h1

    if roc_type == 1:
        thresholds = decision_stats
    elif roc_type == 2:
        thresholds = np.linspace(np.min(decision_stats), np.max(decision_stats), num=99)
    elif roc_type == 3:
        if len(decision_stats) <= 99:
            thresholds = decision_stats
        else:
            n = len(decision_stats) // 99
            thresholds = [decision_stats[i] for i in range(len(decision_stats)) if i % n == 0]
    elif roc_type == 4:
        thresholds = decision_stats[classes == 0]
    elif roc_type == 5:
        h0_decision_stats = decision_stats[classes == 0]
        if len(h0_decision_stats) <= 99:
            thresholds = h0_decision_stats
        else:
            n = len(h0_decision_stats) // 99
            thresholds = [h0_decision_stats[i] for i in range(len(h0_decision_stats)) if i % n == 0]

    thresholds = np.concatenate(([np.min(decision_stats) - 1], thresholds, [np.max(decision_stats) + 1]), axis=0)

    if verbose:
        print('Decision Statistics:\n', decision_stats)
        print('Number of Decision Statistics: ', len(decision_stats), end='\n\n')
        print('Thresholds:\n', thresholds)
        print('Number of Threshold Values: ', len(thresholds), end='\n\n')

    p_d_list = []
    p_fa_list = []
    for threshold in thresholds:
        predictions = decision_stats > threshold
        classes_of_true_predictions = classes[predictions]
        detections = np.sum(classes_of_true_predictions)
        false_alarms = len(classes_of_true_predictions) - detections

        p_fa = false_alarms / num_h0
        p_d = detections / num_h1

        p_fa_list.append(p_fa)
        p_d_list.append(p_d)

    plt.plot(p_fa_list, p_d_list, label=label)

    plt.title(plot_title)
    plt.xlabel('$P_{FA}$')
    plt.ylabel('$P_D$')
    plt.legend(loc='best')
    plt.grid(True)
    if show:
        plt.show()

    return p_fa_list, p_d_list
