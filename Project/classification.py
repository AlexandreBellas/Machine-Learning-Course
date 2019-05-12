


def get_x_and_y(features_dataset):
    x = []
    y = []
    for id, person in zip( range(len(features_dataset)), features_dataset):
        for face_feature in person:
            x.append(face_feature)
            y.append(id)

    return x, y
