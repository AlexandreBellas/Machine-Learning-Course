from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns


def get_x_and_y(features_dataset):
    x = []
    y = []
    for id, person in zip( range(len(features_dataset)), features_dataset):
        for face_feature in person:
            x.append(face_feature)
            y.append(id)

    return np.array(x), np.array(y)


def get_best_knn(x, y, database_name):

    # Parameters to be tested
    ks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Use stratified K-fold to compute the accuracy for each model
    skf = StratifiedKFold(n_splits=10)
    knn_accs = []

    for k in ks:

        accs = []
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]


            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            accs.append(accuracy_score(y_test, knn.predict(x_test)))

        knn_accs.append(np.mean(accs))

    plt.figure(figsize=(15,10))
    plt.plot(ks, knn_accs)
    plt.title("Variação da Acurácia no Modelo KNN para diferentes valores de K para database %s" % database_name, fontsize=16)
    plt.ylabel("Acurácia", fontsize=12)
    plt.xlabel("K", fontsize=12)
    plt.xticks(ks)
    plt.grid(True)
    # plt.show(True)
    plt.savefig("graphs/knn_accuracy.png")

    return ks[np.argmax(knn_accs)]


def get_model_stats(x, y, model, model_name, database_name):
    # Use stratified K-fold to compute the stats for each model
    skf = StratifiedKFold(n_splits=10)


    classes_number = len(np.unique(y))
    model_cm = np.zeros((classes_number, classes_number), dtype=int)
    classes_name = range(0, classes_number)


    accs = []
    precision = []
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        # Compute confusion matrix, accuracy and precision for this model
        model_cm += confusion_matrix(y_test, y_pred, classes_name)
        accs.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='micro'))

    model_acc = np.mean(accs)
    model_precision = np.mean(precision)
    print("Database: %s, Modelo: %s" % (database_name, model_name))
    print("\tAcurácia: %.2f" % model_acc)
    print("\tPrecisão: %.2f" % model_precision)

    plt.figure(figsize=(15,10))
    plt.title("Matriz de Confusão para o modelo %s e database %s" % (model_name, database_name), fontsize=16)
    ax = sns.heatmap(model_cm, annot=True, cbar=False)
    ax.set(xlabel='Classe Predita', ylabel='Verdadeira Classe')
    # plt.show(True)
    plt.savefig("graphs/matriz_confusão_%s_%s" % (model_name, database_name))



