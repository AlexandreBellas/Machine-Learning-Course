from dataset import get_faces, enhance_data, save_faces
from features import get_hog, get_dataset_hogs, save_hogs_dataset, read_hogs_dataset
from classification import get_x_and_y, get_best_knn, get_model_stats, get_best_mlp, get_PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

import random

random.seed(0)

print("================================ PRE PROCESSING ================================")
print("Lendo dados ORL Faces")
orl_faces = get_faces("./datasets/OrlFaces20")
print("ORL FACES:")
print("\tNúmero de pessoas: %d" % len(orl_faces))
print("\tFaces por pessoas: %s" % [len(person) for person in orl_faces])

icmc_faces = get_faces("./datasets/PessoasICMC")
print("Pessoas ICMC:")
print("\tNúmero de pessoas: %d" % len(icmc_faces))
print("\tFaces por pessoas: %s" % [len(person) for person in icmc_faces])

print("---- Gerando faces extras para base Pessoas ICMC ----")
augmented_icmc_faces = enhance_data(icmc_faces)
print("Pessoas ICMC - Extendido:")
print("\tNúmero de pessoas: %d" % len(augmented_icmc_faces))
print("\tFaces por pessoas: %s" % [len(person) for person in augmented_icmc_faces])

print("Salvando Pessoas ICMC - Extendido")
save_faces(augmented_icmc_faces, './datasets/Augmented')

print("================================ HOG EXTRACTION ================================")
read_from_file = False
if read_from_file:
    print("Lendo HOG Features geradas anteriormente")
    icmc_hogs = read_hogs_dataset('icmc_hogs.npy')
    orl_hogs = read_hogs_dataset('orl_hogs.npy')

else:
    print("Gerando novas HOG features")
    icmc_hogs = get_dataset_hogs(augmented_icmc_faces, (32,32))
    save_hogs_dataset(icmc_hogs, 'icmc_hogs')


    orl_hogs = get_dataset_hogs(orl_faces, (12,12))
    save_hogs_dataset(orl_hogs, 'orl_hogs')

print("=============================== TRAINING SECTION ===============================")

x_db = []
y_db = []
database = []

x_orl, y_orl = get_x_and_y(orl_hogs)
x_icmc, y_icmc = get_x_and_y(icmc_hogs)

database_name_orl = "ORL"
database_name_icmc = "ICMC FACES"

x_db.append(x_orl)
x_db.append(x_icmc)
y_db.append(y_orl)
y_db.append(y_icmc)
database.append(database_name_orl)
database.append(database_name_icmc)

print("Dimensão Features Originais: ")
print("\tORL:", len(x_db[0][0]))
print("\tICMC FACES:", len(x_db[1][0]))

for x, y, database_name in zip(x_db, y_db, database):
	print("\n=============================")
	print("DATABASE: %s" % database_name)
	print("=============================")
	print("===================================== KNN  =====================================")


	best_k, best_knn_acc = get_best_knn(x, y, database_name)
	# best_k = 1
	print("Melhor valor para k: %d com acurácia %2.2f%%" % (best_k, best_knn_acc*100))

	best_knn = KNeighborsClassifier(n_neighbors=best_k)
	knn_description = "KNN com k = %s" % best_k

	get_model_stats(x, y, best_knn, knn_description, database_name)


	print("===================================== MLP  =====================================")
	tolerance = 1e-1
	activation = 'logistic'

	best_learning_rate, best_momentum, best_layer1_size, best_layer2_size, best_acc = get_best_mlp(x, y, database_name)

	print("Melhor configuração para MLP")
	print("\tMelhor learning_rate: ", best_learning_rate)
	print("\tMelhor momentum: ", best_momentum)
	print("\tMelhor layer 1 size: ", best_layer1_size)
	print("\tMelhor layer 2 size: ", best_layer2_size)


	layers_cfg = [best_layer1_size]
	if best_layer2_size > 0:
	    layers_cfg.append(best_layer2_size)


	best_mlp = MLPClassifier(hidden_layer_sizes=layers_cfg, solver='sgd', momentum=best_momentum, tol=tolerance,
	                         max_iter=200, random_state=1, learning_rate='adaptive', learning_rate_init=best_learning_rate,
	                         activation=activation)

	mlp_description = "MLP: Learning Rate: %.4f, Momentum: %.2f, Layer 1 Size: %d, Layer 2 Size: %d" %\
	                  (best_learning_rate, best_momentum, best_layer1_size, best_layer2_size)

	get_model_stats(x, y, best_mlp, mlp_description, database_name)

	print("===================================== PCA  =====================================")
	principal_components = get_PCA(x)


	get_model_stats(principal_components, y, best_knn, knn_description, database_name + " PCA")
	get_model_stats(principal_components, y, best_mlp, mlp_description, database_name + " PCA")
