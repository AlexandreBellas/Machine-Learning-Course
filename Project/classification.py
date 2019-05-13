from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import pandas as pd

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


def get_best_mlp(x, y, database_name):
    # layer1_sizes = [10, 20]
    layer1_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    momentum = 0.4
    learning_rate = 0.001

    # Score to store the accuracy for each configuration
    # [[Layer 1 Size, Layer 2 Size, Accuracy]]
    scores = []


    # 1 layer MLP
    # Use stratified K-fold to compute the accuracy for each model
    skf = StratifiedKFold(n_splits=10)
    mlp1_accs = []

    for layer1_size in layer1_sizes:
        model_description = "Layer 1 Size: %d" % layer1_size
        print("training model: %s" % (model_description))

        accs = []
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            mlp = MLPClassifier(hidden_layer_sizes=(layer1_size), solver='sgd', momentum=momentum, tol=1e-4, max_iter=1000, random_state=1, learning_rate='constant', learning_rate_init=learning_rate)
            mlp.fit(x_train, y_train)
            accs.append(accuracy_score(y_test, mlp.predict(x_test)))

        acc = np.mean(accs)
        mlp1_accs.append(acc)
        scores.append([layer1_size, 0, acc])


    plt.figure(figsize=(15,10))
    plt.title("Variação da acurácia em relação ao tamanho da primeira camada para um MLP de uma camada", fontsize=16)
    plt.xlabel("Tamanho da primeira camada", fontsize=12)
    plt.ylabel("Acurácia", fontsize=12)

    plt.grid(True)
    plt.plot(layer1_sizes, mlp1_accs)

    # plt.show(True)
    plt.savefig("graphs/mlp_1_layer_accs.png")

    ## 2 Layers MLP
    # layer1_sizes = [1, 5]
    layer1_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # layer2_sizes = [1, 5]
    layer2_sizes = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    mlp2_accs = dict()

    for layer1_size in layer1_sizes:
        model_description = "Layer 1 Size: %d" % layer1_size
        layer1_accs = []
        for layer2_size in layer2_sizes:
            print("training model: %s, Layer 2 Size: %d" % (model_description, layer2_size))

            # Accuracy for a fixed layer 1 size, varying the size of the second layer
            accs = []
            for train_index, test_index in skf.split(x, y):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]

                mlp = MLPClassifier(hidden_layer_sizes=(layer1_size, layer2_size), solver='sgd', momentum=momentum, tol=1e-4,
                                    max_iter=1000, random_state=1, learning_rate='constant',
                                    learning_rate_init=learning_rate)
                mlp.fit(x_train, y_train)
                accs.append(accuracy_score(y_test, mlp.predict(x_test)))

            acc = np.mean(accs)
            layer1_accs.append(acc)
            scores.append([layer1_size, layer2_size, acc])

        mlp2_accs[model_description] = layer1_accs


    plt.figure(figsize=(15, 10))
    plt.title("Variação da acurácia em relação ao tamanho da segunda camada para um MLP de duas camadas", fontsize=16)
    plt.xlabel("Tamanho da segunda camada", fontsize=12)
    plt.ylabel("Acurácia", fontsize=12)

    for model_description, accs in mlp2_accs.items():
        plt.plot(layer2_sizes, accs, label=model_description)

    plt.legend()
    plt.grid(True)

    # plt.show(True)
    plt.savefig("graphs/mlp_2_layers_accs.png")

    df = pd.DataFrame(data=scores, columns=['Layer 1 size', 'Layer 2 size', 'Accuracy'])
    # print(df)
    df.to_csv("mlp_scores.csv")

    best_cfg = df.iloc[df['Accuracy'].idxmax()]

    return [int(best_cfg[0]), int(best_cfg[1]), best_cfg[2]]



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