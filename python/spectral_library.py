import spectral
import spectral.io.envi as envi
import numpy as np


def get_wavelength(path_image):
    h = envi.read_envi_header(path_image)
    lambdas = (np.asarray(h["wavelength"])).astype(np.float32)
    return lambdas


def read_cube(path_image):
    img = envi.open(path_image)
    arr = img.load()
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    cube = np.zeros((np.shape(arr)[2], np.shape(arr)[0], np.shape(arr)[1]), dtype=np.float32)
    for i in range(np.shape(arr)[2]):
        tmp = np.reshape(arr[:, :, i], (np.shape(arr)[0], np.shape(arr)[1]))
        cube[i, :, :] = tmp

    return cube


def read_cropped_layer(path_image, x_start, y_start, x_end, y_end, layer):
    img = envi.open(path_image)
    cropped_layer = img[y_start:y_end, x_start:x_end, layer]
    cropped_layer = np.reshape(cropped_layer, (cropped_layer.shape[0], cropped_layer.shape[1]))
    cropped_layer[cropped_layer < 0] = 0
    cropped_layer[cropped_layer > 1] = 1
    return cropped_layer


def read_layer(path_image, layer):
    img = envi.open(path_image)
    layer = img.read_band(layer)
    layer[layer < 0] = 0
    layer[layer > 1] = 1
    return layer


def calculate_spectra_rectangle(path_image, x_start, y_start, x_end, y_end):
    img = envi.open(path_image)
    spectra = list()
    n_band = np.shape(img)[2]
    for i in range(n_band):
        sub = img[y_start:y_end, x_start:x_end, i]
        sub = sub[sub <= 1]
        sub = sub[sub >= 0]
        val = np.mean(sub)
        spectra.append(val)

    spectra = np.asarray(spectra)
    spectra = np.reshape(spectra, (1, n_band))
    return spectra


def calculate_spectra_pixel(path_image, x_pix, y_pix):
    img = envi.open(path_image)
    n_band = np.shape(img)[2]
    spectra = img.read_pixel(y_pix, x_pix)
    spectra = np.reshape(spectra, (1, n_band))
    return spectra


def cluster(path_image, nb_cluster, nb_iteration, x_start, y_start, x_end, y_end):
    img = envi.open(path_image)
    sub = img.read_subregion((y_start, y_end), (x_start, x_end))
    sub[sub < 0] = 0
    sub[sub > 1] = 1
    (m, c) = spectral.kmeans(sub, nb_cluster, nb_iteration)
    return (m, c)


def cluster_pca(path_image, nb_cluster, nb_iteration, x_start, y_start, x_end, y_end):
    img = envi.open(path_image)
    sub = img.read_subregion((y_start, y_end), (x_start, x_end))
    sub[sub < 0] = 0
    sub[sub > 1] = 1
    pc = spectral.principal_components(sub)
    pc_0999 = pc.reduce(fraction=0.999)
    sub_pc = pc_0999.transform(sub)
    sub_pc_real = sub_pc.real
    (m, c) = spectral.kmeans(sub_pc_real, nb_cluster, nb_iteration)
    return (m, c)


def calculate_spectral_angles(path_image, x_start, y_start, x_end, y_end):
     img = envi.open(path_image)
     sub = img.read_subregion((y_start, y_end), (x_start, x_end))
     ref_spectra = calculate_spectra_rectangle(path_image, x_start, y_start, x_end, y_end)
     spectral_angles = spectral.spectral_angles(sub, ref_spectra)
     return np.reshape(spectral_angles, (np.shape(spectral_angles)[0], np.shape(spectral_angles)[1]))


def calculate_spectral_map(path_image, x_start, y_start, x_end, y_end):
    img = envi.open(path_image)
    sub = img.read_subregion((y_start, y_end), (x_start, x_end))
    sub[sub < 0] = 0
    sub[sub > 1] = 1
    pc = spectral.principal_components(sub)
    pc_0999 = pc.reduce(fraction=0.999)
    sub = pc_0999.transform(sub)
    sub = sub.real
    spectral_map = np.zeros((np.shape(sub)[0], np.shape(sub)[1]))
    for y in range(1, np.shape(sub)[0]-1):
        for x in range(1, np.shape(sub)[1]-1):
            new_spectra = np.reshape(sub[y, x, :], (1, 1, np.shape(sub)[2]))
            tmp = [
			abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y-1, x-1, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y-1, x, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y-1, x+1, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y, x-1, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y, x+1, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y+1, x-1, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y+1, x, :], (1, np.shape(sub)[2])))),
            abs(spectral.spectral_angles(new_spectra, np.reshape(sub[y+1, x+1,:], (1, np.shape(sub)[2]))))]

            spectral_map[y, x] = np.mean(tmp)

    return spectral_map


# path = 'C:/Users/swift/Desktop/Data/test/SkinAbs G927691_corrected.hdr'
# sa = calculate_spectral_angles(path, 300, 150, 800, 200)
# map = calculate_spectral_angles2(path, 300, 150, 800, 200)
# print(map)
