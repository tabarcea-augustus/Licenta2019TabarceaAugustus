### Models that suppor predict proba ###
#
# from sklearn.utils.testing import all_estimators
# estimators = all_estimators()
# for name, class_ in estimators:
#     if hasattr(class_, 'predict_proba'):
#         print(name)
#
# AdaBoostClassifier
# BaggingClassifier
# BayesianGaussianMixture
# BernoulliNB
# CalibratedClassifierCV
# ComplementNB
# DecisionTreeClassifier
# ExtraTreeClassifier
# ExtraTreesClassifier
# GaussianMixture
# GaussianNB
# GaussianProcessClassifier
# GradientBoostingClassifier
# KNeighborsClassifier
# LabelPropagation
# LabelSpreading
# LinearDiscriminantAnalysis
# LogisticRegression
# LogisticRegressionCV
# MLPClassifier
# MultinomialNB
# NuSVC
# QuadraticDiscriminantAnalysis
# RandomForestClassifier
# SGDClassifier
# SVC
# _BinaryGaussianProcessClassifierLaplace
# _ConstantPredictor
###


import ctypes
import sys
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import SGD, Adam
import numpy as np
import csv
import keras
import progressbar
import json
import gc
import dill as pickle
from functools import partial
import pandas as pd

from os import listdir
from os.path import isfile, join
from keras.layers import Conv3D, MaxPool3D, GlobalMaxPool3D, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import RMSprop
from sklearn.model_selection import ParameterGrid, train_test_split
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from time import sleep
from keras.models import load_model
import multiprocessing
from sklearn.model_selection import StratifiedKFold, KFold
from skmultilearn.model_selection import IterativeStratification
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from src.models.valiml.AdaBoost import AdaBoost
from src.models.valiml.multiclass import OneVsOneClassifier
from src.models.valiml.multiclass import one_vs_rest_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.multiclass import OneVsRestClassifier


# firstDataSetPath = "/home/augt/Public/License2019/clef2019_tuberculosis/data/raw/086e42f6-2077-4d8c-a713-6eddd58a4177_TrainingSet_1_of_2/TrainingSet_1_of_2"
# secondDataSetPath = "/home/augt/Public/License2019/clef2019_tuberculosis/data/raw/07224578-0898-497a-b21d-94fbf883b974_TrainingSet_2_of_2/TrainingSet_2_of_2"


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def loadData(index_mask, use_index_mask=False, single_dimension=False, categorical=True, labels_binary=False,
             loadTestData=False, task2=False):

    resampleFolder = "./data/interim/Normalize01F16MaxPool"

    if loadTestData == True:
        resampleFolder = "./data/processed/Train_Normalize01F16MaxPool"

    files = sorted(listdir(resampleFolder))

    patient_metadata = np.zeros((len(index_mask), 10))

    if not use_index_mask:
        if not task2:
            x_train = np.arange(len(files))
            y_train = np.zeros((len(files),))
            index_mask = list(range(len(files)))
        else:
            x_train = np.arange(len(files))
            y_train = np.zeros((len(files), 6))
            #y_train = np.zeros((len(files),))
            index_mask = list(range(len(files)))
    elif single_dimension:
        if loadTestData:
            x_train = np.zeros((len(index_mask), 256 * 256 * 130 + 10), dtype='float16')
            y_train = np.array(files)
        else:
            if not task2:
                x_train = np.zeros((len(index_mask), 256 * 256 * 130 + 10), dtype='float16')
                y_train = np.zeros((len(index_mask),), dtype='int')
            else:
                x_train = np.zeros((len(index_mask), 256 * 256 * 130 + 10), dtype='float16')
                y_train = np.zeros((len(index_mask), 6), dtype='int')
                #y_train = np.zeros((len(index_mask),), dtype='int')

    else:
        if not task2:
            x_train = np.zeros((len(index_mask), 256, 256, 130, 1), dtype='float16')
            y_train = np.zeros((len(index_mask),), dtype='int')
        else:
            x_train = np.zeros((len(index_mask), 256, 256, 130, 1), dtype='float16')
            y_train = np.zeros((len(index_mask), 6), dtype='int')
            #y_train = np.zeros((len(index_mask),), dtype='int')


    contor = 0
    if use_index_mask:
        bar = progressbar.ProgressBar(max_value=len(index_mask))

    for index in index_mask:
        f = files[index]
        if isfile(join(resampleFolder, f)):
            patientIdentifier = f.replace('.npy', '')

            if not loadTestData:
                with open(
                        './data/raw/8ba42c65-534a-4cdf-b983-e7fcaaaa3002_TrainingSet_metaData.csv', 'r') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    for row in spamreader:
                        if row[0].split(',')[0] == patientIdentifier:
                            if not task2:
                                y_train[contor] = int(row[0].split(',')[11]) - 1
                            else:
                                listaDeBinarizat = []
                                for idx_task2 in range(13,19):
                                    #listaDeBinarizat.append(int(row[0].split(',')[idx_task2]))
                                    y_train[contor, idx_task2-13] = int(row[0].split(',')[idx_task2])
                                    #y_train[contor] = int("".join(str(x) for x in listaDeBinarizat),2)

                            if use_index_mask:
                                for idx_metadata in range(1, 11):
                                    patient_metadata[contor, idx_metadata - 1] = int(row[0].split(',')[idx_metadata])

            if use_index_mask:
                auxImage = np.load(resampleFolder + '/' + f)
                if single_dimension:
                    newImage = auxImage.reshape((-1,))
                    x_train[contor, :newImage.shape[0]] = newImage
                    x_train[contor, newImage.shape[0]:] = patient_metadata[contor]
                else:
                    x_train[contor] = np.expand_dims(auxImage, -1)

            contor += 1
        if use_index_mask:
            bar.update(contor)

    binarisation_label = lambda x: 0 if x < 4 else 1
    vfunc = np.vectorize(binarisation_label)

    if not use_index_mask:
        return x_train, y_train
    elif categorical:
        if single_dimension:
            if labels_binary:
                return x_train, keras.utils.to_categorical(vfunc(y_train), num_classes=2)
            else:
                return x_train, keras.utils.to_categorical(y_train, num_classes=5)
        else:
            if labels_binary:
                return x_train, keras.utils.to_categorical(vfunc(y_train), num_classes=2), patient_metadata
            else:
                return x_train, keras.utils.to_categorical(y_train, num_classes=5), patient_metadata

    else:
        if single_dimension:
            if labels_binary:
                return x_train, vfunc(y_train)
            else:
                return x_train, y_train
        else:
            if labels_binary:
                return x_train, vfunc(y_train), patient_metadata
            else:
                return x_train, y_train, patient_metadata


def buildModel(lr=0.00001, drop_out=0.3, **kwargs):
    K.clear_session()
    K.set_floatx('float16')

    model = Sequential()
    model.add(MaxPool3D((4, 4, 2), input_shape=(256, 256, 130, 1)))
    model.add(BatchNormalization())
    model.add(Conv3D(200, 5))
    model.add(MaxPool3D(2))
    model.add(BatchNormalization())
    model.add(Conv3D(300, 5))
    model.add(MaxPool3D(2))
    model.add(BatchNormalization())
    model.add(Conv3D(350, 3))
    model.add(MaxPool3D(2))
    model.add(BatchNormalization())
    model.add(Conv3D(64, 3))
    model.add(MaxPool3D(2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(5e-5)))
    model.add(Dropout(drop_out))
    model.add(Dense(5, activation='softmax'))
    sgd = Adam(lr=lr)

    model.compile(optimizer=sgd,  # 'rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def save_resources(model, history):
    with open('./models/history.json', 'w') as json_writer:
        json.dump(history, json_writer, sort_keys=True, indent=4,
                  cls=NumpyEncoder)

    print("Saved model to disk")
    print("###################")


def load_resources():
    model = load_model('./models/model.h5')
    print("Loaded model from disk")

    with open('./models/history.json', 'r') as json_reader:
        history = json.load(json_reader)

    return model, history


def run(q, index_train, params, model_file):
    x_train, y_train = loadData(index_train, use_index_mask=True)
    model = buildModel(**params)
    callback_lr = ReduceLROnPlateau(monitor=params['plateau_monitor'], patience=params['plateau_patience'],
                                    cooldown=params['cooldown'], factor=params['factor'], verbose=1,
                                    min_delta=params['plateau_min_delta'])

    callback_stop = EarlyStopping(monitor=params['stop_monitor'], restore_best_weights=False, verbose=1,
                                  min_delta=params['stop_min_delta'], patience=params['stop_patience'])
    history = model.fit(x_train, y_train, epochs=params['epochs'], batch_size=2,
                        validation_split=params['validation_split'], verbose=1, callbacks=[callback_lr, callback_stop])
    model.save(model_file)
    q.put(json.dumps(history.history, cls=NumpyEncoder))


def test(idx, index_train, index_test, params, history, model_file):
    K.clear_session()
    K.set_floatx('float16')
    model = load_model(model_file)
    x_test, y_test = loadData(index_test, use_index_mask=True)
    y_pred = model.predict(x_test, batch_size=2)
    score = model.evaluate(x_test, y_test, batch_size=2)
    try:
        roc_score = roc_auc_score(y_test, y_pred)
    except:
        roc_score = "Nan"

    with open(join('./models', f'{idx}.json'), 'w') as writer:
        json.dump(
            {'params': params, 'test_loss': score[0], 'roc': roc_score, 'test_accuracy': score[1], 'history': history,
             'index_train': index_train, 'index_test': index_test}, writer, sort_keys=True, indent=4,
            cls=NumpyEncoder)


def run1():
    parameters = json.load(open('./src/models/parameters.json', 'r'))

    index, labels = loadData([], use_index_mask=False)
    index_train, index_test = train_test_split(index, stratify=labels, train_size=0.65, test_size=0.35, shuffle=True)

    q = multiprocessing.Queue()
    parameter_list = ParameterGrid(parameters)
    for idx, params in enumerate(parameter_list):
        print(f'Using parameter configuration {idx} of {len(parameter_list)}')
        print(json.dumps(params, indent=4, sort_keys=True))

        model_file = './models/model.h5'
        process = multiprocessing.Process(target=run, args=(q, index_train, params, model_file))
        process.start()
        history = json.loads(q.get())
        process.join()

        process = multiprocessing.Process(target=test, args=(idx, index_train, index_test, params, history, model_file))
        process.start()
        process.join()


def run2():
    params = json.load(open('./src/models/final_parameters.json', 'r'))

    index, labels = loadData([], use_index_mask=False)

    q = multiprocessing.Queue()
    cv = StratifiedKFold(n_splits=4, shuffle=True)
    for idx, (train_index, test_index) in enumerate(cv.split(index, labels)):
        index_train = index[train_index]
        index_test = index[test_index]

        print(json.dumps(params, indent=4, sort_keys=True))

        model_file = './models/model.h5'
        process = multiprocessing.Process(target=run, args=(q, index_train, params, model_file))
        process.start()
        history = json.loads(q.get())
        process.join()

        process = multiprocessing.Process(target=test, args=(idx, index_train, index_test, params, history, model_file))
        process.start()
        process.join()


def run3():
    params = json.load(open('./src/models/final_parameters.json', 'r'))

    index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)
    print(index)
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    for idx, (train_index, test_index) in enumerate(cv.split(index, labels)):

        index_train = index[train_index]
        x_train, y_train = loadData(index_train, use_index_mask=True, single_dimension=True, categorical=False)

        index_test = index[test_index]
        x_test, y_test = loadData(index_test, use_index_mask=True, single_dimension=True, categorical=False)

        print('Training!')
        # classifier = OneVsOneClassifier(AdaBoost(n_estimators=1600, split_criterion='best', mode='exponential', n_random_features=10))
        classifier = AdaBoost(n_estimators=1600, type='real', split_criterion='best', mode='exponential',
                              n_random_features=10000)
        # classifier = GaussianNB()
        # classifier = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1600, algorithm='SAMME.R')
        # classifier = RandomForestClassifier(n_estimators=1000, oob_score=True)
        classifier.fit(x_train, y_train, verbose=True, validation_split=(x_test, y_test))

        print(f'Training acc: {accuracy_score(y_train, classifier.predict(x_train))}')
        pickle.dump(classifier, open(f'models/AdaBoostReal_{idx}.pickle', 'wb'))

        pickle.dump(train_index, open(f"models/train_{idx}.pickle", "wb"))
        pickle.dump(test_index, open(f"models/test_{idx}.pickle", "wb"))

        roc_auc_scores = []

        for class_val, x, y in one_vs_rest_split(x_test, y_test):
            y_pred = (classifier.predict(x) == class_val) * 2 - 1
            roc_auc = roc_auc_score(y, y_pred)
            roc_auc_scores.append(roc_auc)

        y_pred = classifier.predict(x_test)
        matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Roc auc: {roc_auc_scores}')
        print(f'Accuracy: {accuracy}')
        print(f'Matrix: {matrix}')


def test1():
    classifiers_number = 5
    accVecTrain = np.zeros((classifiers_number,))
    rocVecTrain = np.zeros((classifiers_number,))

    for idx in range(0, 5):
        index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)

        x_train, y_train = loadData(index, use_index_mask=True, single_dimension=True, categorical=False)

        trained_test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        trained_train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))

        trained_indexes = trained_train_indexes
        x_train = np.array([x_train[i] for i in trained_indexes])
        y_train = np.array([y_train[i] for i in trained_indexes])
        labels = np.array([labels[i] for i in trained_indexes])

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))
        probas = classifier.predict_proba(x_train)[:, 3:5].sum(axis=1)

        print("\n")
        # for proba in probas:
        #     print ("{:.8f}".format(float(proba)))

        binarization = lambda t: 0 if t < 0.5 else 1
        vfunc = np.vectorize(binarization)
        probas = vfunc(probas)

        binarization2 = lambda t: 0 if t < 4 else 1
        vfunc2 = np.vectorize(binarization2)
        labels = vfunc2(labels)

        # print(f'Training acc: {accuracy_score(labels, probas)}')

        if np.count_nonzero(labels == 0) != len(labels):
            roc_auc = roc_auc_score(labels, probas)
            # print(roc_auc)
            rocVecTrain[idx] = roc_auc

        accVecTrain[idx] = accuracy_score(labels, probas)

    classifiers_number = 5
    accVecTest = np.zeros((classifiers_number,))
    rocVecTest = np.zeros((classifiers_number,))

    for idx in range(0, 5):
        index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)

        x_train, y_train = loadData(index, use_index_mask=True, single_dimension=True, categorical=False)

        trained_test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        trained_train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))

        trained_indexes = trained_test_indexes
        x_train = np.array([x_train[i] for i in trained_indexes])
        y_train = np.array([y_train[i] for i in trained_indexes])
        labels = np.array([labels[i] for i in trained_indexes])

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))
        probas = classifier.predict_proba(x_train)[:, 3:5].sum(axis=1)

        print("\n")
        # for proba in probas:
        #     print ("{:.8f}".format(float(proba)))

        binarization = lambda t: 0 if t < 0.5 else 1
        vfunc = np.vectorize(binarization)
        probas = vfunc(probas)

        binarization2 = lambda t: 0 if t < 4 else 1
        vfunc2 = np.vectorize(binarization2)
        labels = vfunc2(labels)

        # print(f'Training acc: {accuracy_score(labels, probas)}')

        if np.count_nonzero(labels == 0) != len(labels):
            roc_auc = roc_auc_score(labels, probas)
            # print(roc_auc)
            rocVecTest[idx] = roc_auc

        accVecTest[idx] = accuracy_score(labels, probas)

    ####################################################################################################################
    classifiers_number = 5
    accVecTest = np.zeros((classifiers_number,))
    rocVecTest = np.zeros((classifiers_number,))

    for idx in range(0, 5):
        index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)

        x_train, y_train = loadData(index, use_index_mask=True, single_dimension=True, categorical=False)

        trained_test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        trained_train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))

        trained_indexes = trained_test_indexes
        x_train = np.array([x_train[i] for i in trained_indexes])
        y_train = np.array([y_train[i] for i in trained_indexes])
        labels = np.array([labels[i] for i in trained_indexes])

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))
        probas = classifier.predict_proba(x_train)[:, 3:5].sum(axis=1)

        print("\n")
        # for proba in probas:
        #     print ("{:.8f}".format(float(proba)))

        binarization = lambda t: 0 if t < 0.5 else 1
        vfunc = np.vectorize(binarization)
        probas = vfunc(probas)

        binarization2 = lambda t: 0 if t < 4 else 1
        vfunc2 = np.vectorize(binarization2)
        labels = vfunc2(labels)

        # print(f'Training acc: {accuracy_score(labels, probas)}')

        if np.count_nonzero(labels == 0) != len(labels):
            roc_auc = roc_auc_score(labels, probas)
            # print(roc_auc)
            rocVecTest[idx] = roc_auc

        accVecTest[idx] = accuracy_score(labels, probas)

    table = pd.DataFrame(columns=['Split No', 'Train Accuracy', 'Train Roc', 'Test Accuracy', 'Test Roc'])

    for idx in range(0, len(accVecTrain)):
        # df2 = pd.DataFrame([idx, accVecTrain[idx], rocVecTrain[idx], accVecTest[idx], rocVecTest[idx]])
        # table = table.append(df2)
        table.loc[len(table)] = [idx, accVecTrain[idx], rocVecTrain[idx], accVecTest[idx], rocVecTest[idx]]

    print(table)


def test0():
    classifiers_number = 5
    accVec = np.zeros((classifiers_number,))
    rocVec = np.zeros((classifiers_number,))

    for idx in range(0, 5):
        index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)

        x_train, y_train = loadData(index, use_index_mask=True, single_dimension=True, categorical=False)

        trained_test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        trained_train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))

        trained_indexes = trained_train_indexes
        x_train = np.array([x_train[i] for i in trained_indexes])
        y_train = np.array([y_train[i] for i in trained_indexes])
        labels = np.array([labels[i] for i in trained_indexes])

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))
        probas = classifier.predict_proba(x_train)[:, 3:5].sum(axis=1)

        print("\n")
        # for proba in probas:
        #     print ("{:.8f}".format(float(proba)))

        binarization = lambda t: 0 if t < 0.5 else 1
        vfunc = np.vectorize(binarization)
        probas = vfunc(probas)

        binarization2 = lambda t: 0 if t < 4 else 1
        vfunc2 = np.vectorize(binarization2)
        labels = vfunc2(labels)

        # print(f'Training acc: {accuracy_score(labels, probas)}')

        if np.count_nonzero(labels == 0) != len(labels):
            roc_auc = roc_auc_score(labels, probas)
            # print(roc_auc)
            rocVec[idx] = roc_auc

        accVec[idx] = accuracy_score(labels, probas)

    if classifiers_number >= 8:
        for idx in range(0, 3):
            index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False)

            x_train, y_train = loadData(index, use_index_mask=True, single_dimension=True, categorical=False)

            classifier = pickle.load(open(f'models/AdaBoostModels/Ada_Real/AdaBoostReal_{idx}.pickle', 'rb'))
            probas = classifier.predict_proba(x_train)[:, 3:5].sum(axis=1)

            print("\n")
            # for proba in probas:
            #     print ("{:.8f}".format(float(proba)))

            binarization = lambda t: 0 if t < 0.5 else 1
            vfunc = np.vectorize(binarization)
            probas = vfunc(probas)

            binarization2 = lambda t: 0 if t < 4 else 1
            vfunc2 = np.vectorize(binarization2)
            labels = vfunc2(labels)

            # print(f'Training acc: {accuracy_score(labels, probas)}')

            if np.count_nonzero(labels == 0) != len(labels):
                roc_auc = roc_auc_score(labels, probas)
                # print(roc_auc)
                rocVec[idx + 5] = roc_auc

            accVec[idx + 5] = accuracy_score(labels, probas)

    print("##############################################")
    print('Accuracy vector: ', accVec)
    print('Accuracy mean: ' + " ", accVec.mean())
    print('MAX_ACC: ', np.max(accVec))
    print('MIN_ACC: ', np.min(accVec))

    print('ROC_AUC vector; ', rocVec)
    print('ROC_AUC mean: ' + " ", rocVec.mean())
    print('MAX_ROC_AUC: ', np.max(rocVec))
    print('MIN_ROC_AUC: ', np.min(rocVec))
    print("##############################################")


def test2():
    table = pd.DataFrame(
        columns=['Split No', 'Train Accuracy', 'Train Roc', 'Train Acc Low High', 'Test Accuracy', 'Test Roc',
                 'Test Acc Low High'])

    for idx in range(0, 5):
        indexes, _ = loadData([], use_index_mask=False, single_dimension=True, categorical=False)
        x, y = loadData(indexes, use_index_mask=True, single_dimension=True, categorical=False)

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))

        ## Train ##

        train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))
        x_train = x[train_indexes]
        y_train = y[train_indexes]
        y_train_low_high = (y_train < 3).astype('int')

        y_train_pred_proba = classifier.predict_proba(x_train)
        y_train_pred_proba_low_high = y_train_pred_proba[:, :3].sum(axis=1)

        for i in range(0, len(y_train_pred_proba_low_high)):
            print(i, "{:.8f}".format(float(y_train_pred_proba_low_high[i])))

        y_train_pred = y_train_pred_proba.argmax(axis=1)
        y_train_pred_low_high = (y_train_pred_proba_low_high > 0.5).astype('int')

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_roc_auc_low_high = roc_auc_score(y_train_low_high, y_train_pred_proba_low_high)
        train_accuracy_low_high = accuracy_score(y_train_low_high, y_train_pred_low_high)

        print()

        ## Test ##

        test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        x_test = x[test_indexes]
        y_test = y[test_indexes]
        y_test_low_high = (y_test < 3).astype('int')

        y_test_pred_proba = classifier.predict_proba(x_test)
        y_test_pred_proba_low_high = y_test_pred_proba[:, :3].sum(axis=1)

        for j in range(0 + len(y_train_pred_proba_low_high),
                       len(y_test_pred_proba_low_high) + len(y_train_pred_proba_low_high)):
            print(j, "{:.8f}".format(float(y_test_pred_proba_low_high[j - len(y_train_pred_proba_low_high)])))

        y_test_pred = y_test_pred_proba.argmax(axis=1)
        y_test_pred_low_high = (y_test_pred_proba_low_high > 0.5).astype('int')

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_roc_auc_low_high = roc_auc_score(y_test_low_high, y_test_pred_proba_low_high)
        test_accuracy_low_high = accuracy_score(y_test_low_high, y_test_pred_low_high)

        table.loc[len(table)] = [idx, train_accuracy, train_roc_auc_low_high, train_accuracy_low_high, test_accuracy,
                                 test_roc_auc_low_high, test_accuracy_low_high]

    print(table)


def predict(classifier_number):
    indexes, _ = loadData([], use_index_mask=False, single_dimension=True, categorical=False, loadTestData=True)
    x, y = loadData(indexes, use_index_mask=True, single_dimension=True, categorical=False, loadTestData=True)

    classifier = pickle.load(open(f'models/AdaBoostReal_{classifier_number}.pickle', 'rb'))

    ## Predict ##
    y_test_pred_proba = classifier.predict_proba(x)
    y_test_pred_proba_low_high = y_test_pred_proba[:, :3].sum(axis=1)

    with open('src/models/SVRab.txt', 'w') as fd:
        for idx in range(0, len(y_test_pred_proba_low_high)):
            print(y[idx].split('.')[0] + ',' + "{:.8f}".format(float(y_test_pred_proba_low_high[idx])) + '\n')
            fd.write(y[idx].split('.')[0] + ',' + "{:.8f}".format(float(y_test_pred_proba_low_high[idx])) + '\n')


def run1_2():

    index, labels = loadData([], use_index_mask=False, single_dimension=True, categorical=False, task2=True)

    # d = dict([(y, x + 1) for x, y in enumerate(sorted(set(labels)))])
    # labels = np.array([d[x] for x in labels])
    # labels = np.array([x - 1 for x in labels])
    # pickle.dump(d, open(f"models/labels_dictionary_.pickle", "wb"))
    print(index)
    print(labels)
    cv = KFold(n_splits=5, shuffle=True)
    for idx, (train_index, test_index) in enumerate(cv.split(index, labels)):

        index_train = index[train_index]
        x_train, y_train = loadData(index_train, use_index_mask=True, single_dimension=True, categorical=False, task2=True)

        # d = dict([(y, x + 1) for x, y in enumerate(sorted(set(y_train)))])
        # y_train = np.array([d[x] for x in y_train])
        # y_train = np.array([x - 1 for x in y_train])
        # pickle.dump(d, open(f"models/train_dictionary_{idx}.pickle", "wb"))

        index_test = index[test_index]
        x_test, y_test = loadData(index_test, use_index_mask=True, single_dimension=True, categorical=False, task2=True)

        # d = dict([(y, x + 1) for x, y in enumerate(sorted(set(y_test)))])
        # y_test = np.array([d[x] for x in y_test])
        # y_test = np.array([x - 1 for x in y_test])
        # pickle.dump(d, open(f"models/test_dictionary_{idx}.pickle", "wb"))

        print('Training!')
        classifier = OneVsRestClassifier(AdaBoost(n_estimators=2, type='real', split_criterion='best', mode='exponential',
                              n_random_features=1000, verbose=True))
        classifier.fit(x_train, y_train)#, verbose=True, validation_split=(x_test, y_test))

        pickle.dump(classifier, open(f'models/AdaBoostReal_{idx}.pickle', 'wb'))

        pickle.dump(train_index, open(f"models/train_{idx}.pickle", "wb"))
        pickle.dump(test_index, open(f"models/test_{idx}.pickle", "wb"))

        print(f'Training acc: {accuracy_score(y_train, classifier.predict_proba(x_train))}')

        roc_auc_scores = []

        for class_val, x, y in one_vs_rest_split(x_test, y_test):
            y_pred = (classifier.predict_proba(x) == class_val) * 2 - 1
            roc_auc = roc_auc_score(y, y_pred)
            roc_auc_scores.append(roc_auc)

        y_pred = classifier.predict_proba(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Roc auc: {roc_auc_scores}')
        print(f'Accuracy: {accuracy}')


def test2_1():
    table = pd.DataFrame(
        columns=['Split No', 'Train Accuracy', 'Test Accuracy'])

    for idx in range(0, 5):
        indexes, _ = loadData([], use_index_mask=False, single_dimension=True, categorical=False, task2=True)
        x, y = loadData(indexes, use_index_mask=True, single_dimension=True, categorical=False, task2=True)

        classifier = pickle.load(open(f'models/AdaBoostReal_{idx}.pickle', 'rb'))

        ## Train ##

        train_indexes = pickle.load(open(f'models/train_{idx}.pickle', 'rb'))
        x_train = x[train_indexes]
        y_train = y[train_indexes]

        y_train_pred = classifier.predict_proba(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        ## Test ##

        test_indexes = pickle.load(open(f'models/test_{idx}.pickle', 'rb'))
        x_test = x[test_indexes]
        y_test = y[test_indexes]

        y_test_pred = classifier.predict(x_test)


        test_accuracy = accuracy_score(y_test, y_test_pred)

        table.loc[len(table)] = [idx, train_accuracy, test_accuracy]

    print(table)


def predict2(classifier_number):
    indexes, _ = loadData([], use_index_mask=False, single_dimension=True, categorical=False, loadTestData=True, task2=True)
    x, y = loadData(indexes, use_index_mask=True, single_dimension=True, categorical=False, loadTestData=True, task2=True)

    classifier = pickle.load(open(f'models/AdaBoostReal_{classifier_number}.pickle', 'rb'))

    ## Predict ##
    y_test_pred = classifier.predict_proba(x)

    with open('src/models/CTRlab2.txt', 'w') as fd:
        for idx in range(0, len(y_test_pred)):
            str = y[idx].split('.')[0] + ',' + ",".join(["{:.8f}".format(float(y_test_pred[idx, k])) for k in range(0, 6)]) + '\n'
            print(str)
            fd.write(str)

if __name__ == '__main__':
    # test2()
    predict2(3)
    #test2_1()
    #run1_2()