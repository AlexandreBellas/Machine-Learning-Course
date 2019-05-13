from dataset import get_faces, enhance_data, save_faces
from features import get_hog, get_dataset_hogs, save_hogs_dataset, read_hogs_dataset
from classification import get_x_and_y, get_best_knn, get_model_stats, get_best_mlp

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from skimage import io


orl_faces = get_faces("./datasets/OrlFaces20")
icmc_faces = get_faces("./datasets/PessoasICMC")

print("ORL FACES")
print(len(orl_faces))

for person in orl_faces:
    print(len(person))

#print(person)

# io.imshow(orl_faces[9][9])
# io.show()

print("ICMC FACES")
print(len(icmc_faces))

for person in icmc_faces:
    print(len(person))
    # io.imshow(person[0])
    # io.show()

# io.imshow(orl_faces[9][9])
# io.show()

augmented_icmc_faces = enhance_data(icmc_faces)
print("ENHANCES ICMC FACES")
print(len(icmc_faces))

# for person in augmented_icmc_faces:
#     print(len(person))
    # for face in person:
    #     io.imshow(face)
    #     io.show()


save_faces(augmented_icmc_faces, './datasets/Augmented')

# HOG
read_from_file = True
if read_from_file:
    icmc_hogs = read_hogs_dataset('icmc_hogs.npy')
    orl_hogs = read_hogs_dataset('orl_hogs.npy')

else:
    icmc_hogs = get_dataset_hogs(augmented_icmc_faces)
    save_hogs_dataset(icmc_hogs, 'icmc_hogs')

    orl_hogs = get_dataset_hogs(orl_faces)
    save_hogs_dataset(orl_hogs, 'orl_hogs')

print("True")


# Get X and Y for a feature dataset
x, y = get_x_and_y(orl_hogs)
# print("LEN X: ", len(x))
# print(x)
# print("LEN Y", len(y))
# print(y)

print(x)
# best_k = get_best_knn(x, y, "ORL")
best_k = 1
print("Melhor valor para k: ", best_k)

# best_knn = KNeighborsClassifier(n_neighbors=best_k)
# knn_description = "KNN com k = %s" % best_k
#
# get_model_stats(x, y, best_knn, knn_description, "ORL")

best_layers_cfg = get_best_mlp(x, y, "ORL")
print("Best MLP:", best_layers_cfg)

momentum = 0.4
learning_rate = 0.001

layers_cfg = [best_layers_cfg[0]]
if best_layers_cfg[1] != 0:
    layers_cfg.append(best_layers_cfg[1])


best_mlp = MLPClassifier(hidden_layer_sizes=layers_cfg, solver='sgd', momentum=momentum, tol=1e-4, max_iter=1000, random_state=1, learning_rate='constant', learning_rate_init=learning_rate)

print(best_mlp)


mlp_description = "MLP: Layer 1 Size: %d, Layer 2 Size: %d" % (best_layers_cfg[0], best_layers_cfg[1])
get_model_stats(x, y, best_mlp, mlp_description, "ORL")