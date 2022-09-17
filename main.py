import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from data import load_data
from show_channel_weights import show_channel_weights
import matplotlib.pyplot as plt
from roc import generate_ROC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


strongest_overt_channels = [66, 16, 76, 177, 79, 195, 101, 57, 202, 58, 36, 37, 70, 117, 59, 166, 41, 71, 141, 189, 86,
                            39, 73, 24, 68, 17, 80, 83, 35, 90, 69, 8, 160, 25, 47, 92, 119, 48, 65, 32, 67, 174, 93,
                            127, 77
    , 85, 188, 197, 116, 61, 11, 34, 107, 158, 38, 180, 190, 33, 196, 74, 45, 40, 22, 14, 82, 10, 169, 181, 64, 78, 30,
                            194, 165, 97, 159, 63, 96, 31, 56, 81, 43, 98, 42, 75, 183, 20, 18, 84, 16,
                            7, 178, 94, 162, 102, 105, 60, 72, 46, 135, 62, 132, 193, 5, 21, 175, 171, 125, 134, 150,
                            143, 15, 139, 29, 192, 184, 7, 9, 26, 161, 91, 142, 186, 199, 203, 122, 131, 164, 13, 52,
                            118, 147,
                            88, 12, 133, 95, 3, 6, 1, 148, 87, 176, 106, 110, 19, 115, 99, 129, 109, 201, 0, 104, 156,
                            138, 50, 89, 137, 49, 51, 27, 200, 173, 55, 53, 28, 157, 149, 179, 182, 172, 111, 4, 198,
                            23, 103,
                            2, 163, 126, 191, 54, 152, 146, 114, 145, 44, 144, 113, 130, 108, 123, 153, 121, 170, 168,
                            120, 155, 128, 124, 112, 151, 100, 185, 187, 136, 154, 140]

strongest_imagined_channels = [86, 12, 10, 20, 66, 22, 72, 30, 94, 18, 106, 135, 2, 28, 17, 179, 189, 44, 98, 125, 115,
                               80, 84, 29, 16, 198, 161, 75, 132, 55, 64, 158, 139, 38, 83, 102, 199, 1, 107, 141, 58,
                               88, 164, 165
    , 203, 90, 0, 195, 104, 105, 35, 148, 7, 196, 63, 170, 36, 93, 162, 137, 40, 74, 57, 14, 79, 172, 178, 52, 160, 200,
                               45, 78, 54, 87, 159, 182, 65, 152, 103, 68, 33, 85, 143, 138, 37, 62, 202
    , 190, 134, 51, 96, 49, 11, 6, 19, 111, 171, 191, 73, 147, 124, 185, 8, 32, 193, 100, 42, 99, 5, 194, 61, 129, 201,
                               59, 82, 46, 197, 177, 23, 41, 4, 192, 26, 95, 183, 122, 70, 92, 97, 168, 6,
                               0, 108, 150, 76, 39, 81, 31, 21, 27, 113, 50, 109, 47, 181, 118, 3, 9, 101, 34, 130, 25,
                               110, 116, 131, 163, 13, 112, 48, 67, 24, 53, 56, 91, 69, 71, 142, 89, 144, 184, 77, 149,
                               117, 156, 43
    , 119, 180, 157, 175, 188, 176, 166, 133, 15, 114, 121, 173, 146, 127, 136, 145, 169, 126, 153, 167, 123, 174, 128,
                               186, 120, 154, 155, 151, 140, 187]


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
def fit_svm(X_train, y_train, X_test, y_test, lambda_val=1, kernel='linear'):
    C = 1 / lambda_val

    svc = SVC(kernel=kernel, C=C, max_iter=50000)
    svc.fit(X_train, y_train)
    try:
        channel_weights = svc.coef_[0]
    except:
        channel_weights = np.arange(204)

    y_hat = svc.predict(X_test)
    accuracy = (y_hat == y_test).sum() / len(y_test)
    decision_statistics = svc.decision_function(X_test)

    return accuracy, channel_weights, decision_statistics, len(svc.support_vectors_)


def cross_validation(data, labels, lambda_val_list=[0.01, 1, 100, 10000], num_level1_folds=6, num_level2_folds=5,
                     dataset='Overt', verbose=True, num_channels=204):
    channels = strongest_imagined_channels[-num_channels:]
    data = data[:, channels]
    channel_weights = np.zeros(data.shape[1])

    test_accuracies = []
    best_lambda_vals = []

    cross_validated_decision_stats = np.zeros(labels.shape)
    cross_validated_labels = np.zeros(labels.shape)
    p_fa_lists = []
    p_d_lists = []

    kf1 = KFold(n_splits=num_level1_folds, shuffle=True)

    i = 0
    for train_index, test_index in kf1.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        lambda_val_accuracies = []
        for lambda_val in lambda_val_list:
            lambda_val_accuracy = 0

            kf2 = KFold(n_splits=num_level2_folds)
            for cross_val_train_index, val_index in kf2.split(train_data):
                cross_validation_train_data, val_data = train_data[cross_val_train_index], train_data[val_index]
                cross_validation_train_labels, val_labels = train_labels[cross_val_train_index], train_labels[val_index]

                val_accuracy, _, _, _ = fit_svm(cross_validation_train_data, cross_validation_train_labels, val_data,
                                             val_labels, lambda_val)
                lambda_val_accuracy += val_accuracy

            lambda_val_accuracy /= num_level2_folds
            lambda_val_accuracies.append(lambda_val_accuracy)

        # Use the value of lambda and use entire train set to make predictions for test set
        best_lambda_val_idx = np.argmax(lambda_val_accuracies)
        best_lambda_val = lambda_val_list[best_lambda_val_idx]
        best_lambda_vals.append(best_lambda_val)

        test_accuracy, test_weights, test_decision_statistics, num_support_vecs = fit_svm(train_data, train_labels, test_data,
                                                                        test_labels, best_lambda_val)
        test_accuracies.append(test_accuracy)
        channel_weights += np.abs(test_weights)


        p_fa_list, p_d_list = generate_ROC(test_labels, test_decision_statistics, roc_type=1, plot_title='ROC Curve',
                     label=f'Fold {i}', verbose=False, show=False)

        p_fa_lists.append(p_fa_list)
        p_d_lists.append(p_d_list)
        cross_validated_decision_stats[test_index] = test_decision_statistics
        cross_validated_labels[test_index] = test_labels
        i += 1

    generate_ROC(cross_validated_labels, cross_validated_decision_stats, roc_type=1,
                 plot_title='ROC Curve for Overt Dataset Cross Validation',
                 label='Cross Validated ROC', verbose=False, show=False)

    cross_validation_accuracy = np.sum(test_accuracies) / len(test_accuracies)
    if verbose:
        print('Best Lambda Values:', best_lambda_vals)
        print('Test Accuracies:', test_accuracies)
        print(
            f'Cross Validation Overall Accuracy: {cross_validation_accuracy}')  # All test sets have 40 entries, so the total accuracy is average of all test accuracies
        print(np.mean(channel_weights))
        print(np.var(channel_weights))
    return cross_validation_accuracy


def cross_validation_one_level(data, labels, test_data, test_labels, lambda_val_list=[0.01, 1, 100, 10000],
                               num_level1_folds=6):
    kf1 = KFold(n_splits=num_level1_folds, shuffle=True)
    lambda_val_accuracies = []
    for lambda_val in lambda_val_list:
        lambda_val_accuracy = 0

        for train_index, val_index in kf1.split(data):
            train_data, val_data = data[train_index], data[val_index]
            train_labels, val_labels = labels[train_index], labels[val_index]
            val_accuracy, _, _ = fit_svm(train_data, train_labels, val_data,
                                         val_labels, lambda_val)
            lambda_val_accuracy += val_accuracy

        lambda_val_accuracy /= num_level1_folds
        lambda_val_accuracies.append(lambda_val_accuracy)

    # Use the value of lambda and use entire train set to make predictions for test set
    best_lambda_val_idx = np.argmax(lambda_val_accuracies)
    best_lambda_val = lambda_val_list[best_lambda_val_idx]

    test_accuracy, test_weights, test_decision_statistics = fit_svm(data, labels, test_data, test_labels,
                                                                    best_lambda_val)
    return test_accuracy


def average_pd_values(p_fa_lists, p_d_lists, p_fa_targets=np.linspace(0, 1, num=20)):
    p_d_averages = np.zeros(p_fa_targets.shape)

    for i in range(len(p_fa_lists)):
        p_fa_list, p_d_list = p_fa_lists[i], p_d_lists[i]
        for j, p_fa_target in enumerate(p_fa_targets):
            idx = find_nearest(p_fa_list, p_fa_target)
            p_d_averages[j] += p_d_list[idx]

    p_d_target_values = p_d_averages / len(p_fa_lists)

    p_fa_targets = np.concatenate((np.zeros(1), p_fa_targets))
    p_d_target_values = np.concatenate((np.zeros(1), p_d_target_values))

    return p_fa_targets, p_d_target_values


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if (array[idx] - value) > 0 and idx != 0:
        idx -= 1
    return idx


def plot_channel_weights(channel_weights):
    channel_weights = np.abs(channel_weights)
    show_channel_weights(channel_weights)
    channels = range(len(channel_weights))

    test_weights_sorted, channels_sorted = (list(t) for t in zip(*sorted(zip(channel_weights, channels))))
    print('Strongest Channels: ', channels_sorted[-5:])
    print('Channel Weights: ', test_weights_sorted[-5:], end='\n\n')

    plt.plot(channels, channel_weights)
    plt.title('Channel Weights for 1st Level Cross Validation Fold #1')
    plt.xlabel('Channel Number')
    plt.ylabel('Channel Weight')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    data_imagined, labels_imagined, data_overt, labels_overt = load_data()

    print(cross_validation(data_overt, labels_overt, dataset='Overt'))
    print(cross_validation(data_imagined, labels_imagined, dataset='Imagined'))

    print('---------- OVERT ----------')
    
    for num_channels in [204, 180, 150, 120, 90, 60, 30]:
        ac = 0
        for i in range(10):
            ac += cross_validation(data_overt, labels_overt, dataset='Overt', num_channels=num_channels)
        print(num_channels, ac / 10)

    print('---------- IMAGINED ----------')
    for num_channels in [204, 180, 150, 120, 90, 60, 30]:
        ac = 0
        for i in range(10):
            a = cross_validation(data_imagined, labels_imagined, dataset='Imagined', num_channels=num_channels)
            print(a)
            ac += a
        print(num_channels, ac / 10)


    ac1 = 0
    ac2 = 0
    ac3 = 0
    ac4 = 0
    for i in range(10):
        overt_data_train, overt_data_test, overt_labels_train, overt_labels_test = train_test_split(data_overt,
                                                                                                    labels_overt,
                                                                                                    test_size=0.2)
        imagined_data_train, imagined_data_test, imagined_labels_train, imagined_labels_test = train_test_split(
            data_imagined,
            labels_imagined,
            test_size=0.2)
    
        ac1 += cross_validation_one_level(overt_data_train, overt_labels_train, overt_data_test, overt_labels_test)
        ac2 += cross_validation_one_level(overt_data_train, overt_labels_train, imagined_data_test, imagined_labels_test)
        ac3 += cross_validation_one_level(imagined_data_train, imagined_labels_train, imagined_data_test, imagined_labels_test)
        ac4 += cross_validation_one_level(imagined_data_train, imagined_labels_train, overt_data_test, overt_labels_test)
    
    print('Train Overt, Test Overt')
    print(ac1 / 10)
    
    print('Train Overt, Test Imagined')
    print(ac2 / 10)
    
    print('Train Imagined, Test Imagined')
    print(ac3 / 10)
    
    print('Train Imagined, Test Overt')
    print(ac4 / 10)
