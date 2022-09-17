import numpy as np


DATA_IMAGINED_CLASS1_CSV_PATH = 'data/feaSubEImg_1.csv'
DATA_IMAGINED_CLASS2_CSV_PATH = 'data/feaSubEImg_2.csv'

DATA_OVERT_CLASS1_CSV_PATH = 'data/feaSubEOvert_1.csv'
DATA_OVERT_CLASS2_CSV_PATH = 'data/feaSubEOvert_2.csv'


def load_data():
    data_imagined_class1 = np.genfromtxt(DATA_IMAGINED_CLASS1_CSV_PATH, delimiter=',')
    data_imagined_class2 = np.genfromtxt(DATA_IMAGINED_CLASS2_CSV_PATH, delimiter=',')

    data_overt_class1 = np.genfromtxt(DATA_OVERT_CLASS1_CSV_PATH, delimiter=',')
    data_overt_class2 = np.genfromtxt(DATA_OVERT_CLASS2_CSV_PATH, delimiter=',')

    data_imagined = np.concatenate((data_imagined_class1, data_imagined_class2), axis=1)
    data_overt = np.concatenate((data_overt_class1, data_overt_class2), axis=1)

    labels_imagined = np.concatenate((np.zeros((data_imagined_class1.shape[1])), np.ones((data_imagined_class2.shape[1]))))
    labels_overt = np.concatenate((np.zeros((data_overt_class1.shape[1])), np.ones((data_overt_class1.shape[1]))))

    data_imagined = data_imagined.T
    data_overt = data_overt.T

    return data_imagined, labels_imagined, data_overt, labels_overt


if __name__ == '__main__':
    load_data()