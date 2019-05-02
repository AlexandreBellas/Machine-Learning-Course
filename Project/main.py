from dataset import get_faces, enhance_data, save_faces, get_hog
from skimage import io


orl_faces = get_faces("./datasets/OrlFaces20")
icmc_faces = get_faces("./datasets/PessoasICMC")

print("ORL FACES")
print(len(orl_faces))

for person in orl_faces:
    print(len(person))

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


save_faces(augmented_icmc_faces , './datasets/Augmented')

# HOG
get_hog(augmented_icmc_faces[0][4])