# Shared utility functions for the NTIRE2022 spectral challenges
import hdf5storage
import numpy as np
import cv2 as cv
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from sklearn.cluster import MiniBatchKMeans


from Conf import ANALOG_CHANNEL_GAIN, TYPICAL_SCENE_REFLECTIVITY, MAX_VAL_8_BIT, MAX_VAL_12_BIT


def load_ms_filter(csv):
    df = pd.read_csv(csv, skiprows=[1])
    ms_filter = df.iloc[:, 1:17].to_numpy()
    bands = df['Channel'].to_numpy()

    df = pd.read_csv(csv)
    ms_peaks = df.iloc[0, 1:17].to_numpy().astype(np.int16)

    return ms_filter, bands, ms_peaks

def load_rgb_filter(csv):
    df = pd.read_csv(csv)
    camera_filter = df[['R', 'G1', 'B']].to_numpy() * ANALOG_CHANNEL_GAIN
    bands = df['Wavelength[nm]'].to_numpy()
    return camera_filter, bands


def addPoissonAndDarkNoise(signal, divFactorTo_1PE=1, npe=1):
    """
    Add camera noise, based on Poisson and Gaussian Normal dark noise model.

    :param signal: Input signal. Can be of any numeric type, with any array dimensionality. Default units: [Npe]
    :param darkNoise: dark noise standard deviation. Units: [Npe]
    :param divFactorTo_1PE: (Optional) modifier for converting the input signal to Npe units.
    :param npe: (Optional) The target Npe units. USE ONLY in conjunction with divFactorTo_1PE, or if input image is already in Npe units. 0 means no shot noise

    :return: The signal with added noise (without changing the signal mean). As numpy array
    """
    if npe == 0:
        return signal

    scale = npe / divFactorTo_1PE
    shotNoiseSignal = np.random.poisson(signal.clip(0, None) * scale)  # Clip signal to positive values only. Randomize signal by poisson Shot noise model
    noisySignal = shotNoiseSignal / scale  # Total noisy signal. (scaled to original range)

    return noisySignal


def addNoise(rgb, npe=1, div_factor_to_1npe=1):
    """
    Add camera simulated noise to an Image, based on Poisson and Gaussian Normal dark noise model.

    :param npe: light intensity, determines the noise level
    :param div_factor_to_1npe: division by this factor brings the image to 1 npe

    :return: A new Image with added noise (without changing the scale of the signal)
    """
    noisy_rgb = addPoissonAndDarkNoise(rgb, npe=npe, divFactorTo_1PE=div_factor_to_1npe)

    return noisy_rgb


def make_spectral_bands(nm_start, nm_stop, nm_step, dtype=np.int32):
    """
    boilerplate code to make a uniform spectral wavelength range

    :param nm_start: start wavelength in [nm]
    :param nm_stop:  stop wavelength (inclusive) in [nm]
    :param nm_step: spectral resolution in [nm]
    :param dtype: default - integer

    :return: numpy array of wavelengths
    """
    if nm_step <= 0:
        raise ValueError("make_spectral_bands: step must be positive.")
    return np.arange(start=nm_start,
                     stop=nm_stop + nm_step / 2, # make sure to include the stop wavelength
                     step=nm_step).astype(dtype)



def resampleHSPicked(cube, bands, newBands, interpMode='linear', fill_value='extrapolate'):
    """
    Resample a hyperspectral cube at picked arbitrary 'newBands'

    :param cube: numpy array of HS data, shape [H, W, num_hs_channels] or [num_samples, num of channels]
    :param bands: numpy array of the wavelength (nm) of each channel in the cube, shape [num of channels]
    :param newBands: numpy array of the wavelength (nm) at which to resample the cube, shape [num of new bands]
    :param interpMode: See more details in CubeUtils.resampleHS
    :param fill_value: if 'extrapolate', then data will be extrapolated,
                       if float, then the value will be set at both ends of the range
                       if tuple of floats (a, b), then a will fill the bottom of the range and b the top
                       default is NaN

    :return: a numpy array of the sampled cube, shape [H, W, num of new bands]
    """
    interpModes = ['zero', 'slinear', 'quadratic', 'cubic', 'linear', 'nearest', 'previous', 'next']  # taken from interp1d
    if interpMode not in interpModes:
        raise ValueError(f"resampleHSPicked: {interpMode} is not a valid interpMode. Options are {','.join(interpModes)}.")

    interpfun = interp1d(bands, cube, axis=-1, kind=interpMode, assume_sorted=True, fill_value=fill_value, bounds_error=False)
    resampled = interpfun(newBands)

    return resampled


def projectCube(pixels, filters, clipNegative=False):
    """
    project multispectral pixels to low dimension

    :param pixels: numpy array of multispectral pixels, shape [..., num_hs_bands]
    :param filters: filter response, [num_hs_bands, num_mc_chans]
    :param clipNegative: whether to clip negative values

    :return: a numpy array of the projected pixels, shape [..., num_mc_chans]

    :raise: RuntimeError if `pixels` or `filters` are passed transposed

    """

    # assume the number of spectral channels match (will crash inside if not)
    if np.shape(pixels)[-1] != np.shape(filters)[0]:
        raise RuntimeError(f'{__file__}: projectCube - incompatible dimensions! got {np.shape(pixels)} and {np.shape(filters)}')

    projected = np.matmul(pixels, filters)

    if clipNegative:
        projected = projected.clip(0, None)

    return projected


def projectHS(cube, cube_bands, qes, qe_bands, clipNegative, interp_mode='linear'):
    """
    project a spectral array

    :return: numpy array of projected data, shape [..., num_channels ]
    """
    if not np.all(qe_bands == cube_bands):  # then sample the qes on the data bands
        dx_qes = qe_bands[1] - qe_bands[0]
        dx_hs = cube_bands[1] - cube_bands[0]
        if np.any(np.diff(qe_bands) != dx_qes) or np.any(np.diff(cube_bands) != dx_hs):
            raise ValueError(f'V81Filter.projectHS - can only interpolate from uniformly sampled bands\n'
                             f'got hs bands: {cube_bands}\n'
                             f'filter bands: {qe_bands}')

        if dx_qes < 0:
            # we assume the qe_bands are sorted ascending inside resampleHSPicked, reverse them
            qes = qes[::-1]
            qe_bands = qe_bands[::-1]

        # find the limits of the interpolation, WE DON'T WANT TO EXTRAPOLATE!
        # the limits must be defined by the data bands so the interpolated qe matches
        min_band = cube_bands[
            np.argwhere(cube_bands >= qe_bands.min()).min()]  # the first data band which has a respective qe value
        max_band = cube_bands[
            np.argwhere(cube_bands <= qe_bands.max()).max()]  # the last data band which has a respective qe value
        # TODO is there a minimal overlap we want to enforce?

        cube = cube[..., np.logical_and(cube_bands >= min_band, cube_bands <= max_band)]
        shared_bands = make_spectral_bands(min_band, max_band,
                                           dx_hs)  # shared domain with the spectral resolution of the spectral data
        qes = resampleHSPicked(qes.T, bands=qe_bands, newBands=shared_bands, interpMode=interp_mode,
                               fill_value=np.nan).T

    return projectCube(cube, qes, clipNegative=clipNegative)


def createNoisyRGB(cube, cube_bands, rgb_filter, filter_bands, npe):
    rgb         = projectHS(cube, cube_bands, rgb_filter, filter_bands, clipNegative=True)
    noisy_rgb   = addNoise(rgb, npe=npe)

    return noisy_rgb


def save_jpg(rgb, path, quality):
    # scale to 8bit
    rgb *= (TYPICAL_SCENE_REFLECTIVITY / rgb.mean()) * MAX_VAL_8_BIT
    rgb = rgb.clip(0, MAX_VAL_8_BIT).astype(np.uint8)

    # save to disk
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    cv.imwrite(path, bgr, [cv.IMWRITE_JPEG_QUALITY, quality])

def loadCube(path):
    with h5py.File(path, 'r') as mat:
        cube = np.array(mat['cube']).T
        cube_bands = np.array(mat['bands']).squeeze()
    return cube, cube_bands

def saveCube(path, cube, bands=None, norm_factor=None):
    hdf5storage.write({u'cube': cube,
                       u'bands': bands,
                       u'norm_factor': norm_factor}, '.',
                       path, matlab_compatible=True)

def create_multispectral(cube, cube_bands, ms_filter, ms_filter_bands):
    ms = projectHS(cube, cube_bands, ms_filter, ms_filter_bands, clipNegative=True)

    # make sure pixels can be divided to 4x4 blocks
    h, w = ms.shape[:2]
    s = 4
    h = (h // s) * s
    w = (w // s) * s
    ms = ms[:h, :w] # crop

    # scale to 0,1
    norm_factor = ms.max()
    ms /= ms.max()

    # create mosaic
    mosaic = np.zeros([h, w])
    for i in range(s):
        for j in range(s):
            idx = s * i + j
            mosaic[i::s, j::s] = ms[i::s, j::s, idx]  # mosaic

    # scale to 12bit
    mosaic *= (TYPICAL_SCENE_REFLECTIVITY / mosaic.mean()) * MAX_VAL_12_BIT
    mosaic = mosaic.clip(0, MAX_VAL_12_BIT).astype(np.uint16)

    return mosaic, ms, norm_factor


def compute_mse(a, b):
    """
    Compute the mean squared error between two arrays

    :param a: first array
    :param b: second array with the same shape

    :return: MSE(a, b)
    """
    assert a.shape == b.shape
    diff = a - b
    return np.power(diff, 2)


def compute_rmse(a, b):
    """
    Compute the root mean squared error between two arrays

    :param a: first array
    :param b: second array with the same shape

    :return: RMSE(a, b)
    """
    sqrd_error = compute_mse(a, b)

    return np.sqrt(np.mean(sqrd_error))


def compute_psnr(a, b, peak):
    """
    compute the peak SNR between two arrays

    :param a: first array
    :param b: second array with the same shape
    :param peak: scalar of peak signal value (e.g. 255, 1023)

    :return: psnr (scalar)
    """
    sqrd_error = compute_mse(a, b)
    mse = sqrd_error.mean()
    # TODO do we want to take psnr of every pixel first and then mean?
    return 10 * np.log10((peak ^ 2) / mse)


def flatten_and_normalize(arr):
    h, w, c = arr.shape
    arr = arr.reshape([h * w, c])
    norms = np.linalg.norm(arr, ord=2, axis=-1)
    norms[norms == 0] = 1 # remove zero division problems

    return arr / norms


def compute_sam(a, b):
    """
    spectral angle mapper

    :param a: first array
    :param b: second array with the same shape

    :return: mean of per pixel SAM
    """
    assert a.shape == b.shape
    # normalize each array per pixel, so the dot product will be determined only by the angle
    a = flatten_and_normalize(a)
    b = flatten_and_normalize(b)
    angles = np.dot(a, b)
    sams   = np.arccos(angles)

    return sams.mean()

def computeMRAE(groundTruth, recovered):
    """
    Compute MRAE between two images
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :return: Mean Realative Absolute Error between `recovered` and `groundTruth`.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

    difference = np.abs(groundTruth - recovered) / groundTruth
    mrae = np.mean(difference)

    return mrae


def evalBackProjection(groundTruth, recovered, cameraResponse):
    """
    Score the colorimetric accuracy of a recovered spectral image vs. a ground truth reference image.
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param cameraResponse: camera response functions. (Spectral_Dimension x RGB_Dimension)
    :return: MRAE between ground-truth and recovered RGBs.
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[2] == cameraResponse.shape[0], "Spectral dimension mismatch between spectral images and camera response functions"

    specDim = cameraResponse.shape[0]  # spectral dimension

    # back projection + reshape the data into num_of_samples x spectral_dimensions
    groundTruthRGB = np.matmul(groundTruth.reshape(-1, specDim), cameraResponse)
    recoveredRGB = np.matmul(recovered.reshape(-1, specDim), cameraResponse)

    # calculate MRAE
    difference = np.abs(groundTruthRGB - recoveredRGB) / groundTruthRGB
    mrae = np.mean(difference)

    return mrae


def labelPixelGroup(groundTruth, numberOfGroups=1000):
    """
    Use k-means to group similar spectra, and label the pixels with the group numbers.
    Note that k-means are calculated on normalized spectra (regardless of the intensity level)
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param numberOfGroups: number of representative spectra.
    :return: pixel labels that record to which group the spectrum at each pixel belongs. (Height x Width)
    """

    height, width, specDim = groundTruth.shape

    # reshape the data into num_of_samples x spectral dimensions, and normalize the spectra
    groundTruthList = groundTruth.reshape(-1, specDim)
    normalizedGroundTruthList = groundTruthList / np.linalg.norm(groundTruthList, axis=1, keepdims=True)

    # kmeans calculation best in n_init trials (mini batch kmeans approximation with batch size at 10% image size)
    batchSize = int(height * width * 0.1)
    trials = 5
    kmeans = MiniBatchKMeans(n_clusters=numberOfGroups, batch_size=batchSize, n_init=trials).fit(normalizedGroundTruthList)

    labeledImage = kmeans.labels_
    labeledImage = labeledImage.reshape(height, width)

    return labeledImage


def weightedAccuracy(groundTruth, recovered, labeledImage):
    """
    Compute the mean group performance in MRAE. Spectra are grouped by the ``labelPixelGroup'' function
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param labeledImage: labeled image, output of ``labelPixelGroup'' function. (Height x Width)
    :return: mean group performance in MRAE
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[:2] == labeledImage.shape[:2], "Size not match for spectral and labeled images"

    specDim = groundTruth.shape[2]  # spectral dimension

    # reshape the inputs into num_of_samples x spectral_dimensions
    groundTruthList = groundTruth.reshape(-1, specDim)
    recoveredList = recovered.reshape(-1, specDim)
    labelList = labeledImage.reshape(-1)

    # list of group numbers
    groups = np.sort(np.unique(labelList)).astype(int)

    allMrae = []  # used to collect mrae of all groups

    # group by group calculating mean MRAE
    for groupNum in groups:
        groupPixels = labelList == groupNum

        groupGroundTruth = groundTruthList[groupPixels, :]
        groupRecovered = recoveredList[groupPixels, :]

        # calculate MRAE
        groupDiff = np.abs(groupGroundTruth - groupRecovered) / groupGroundTruth
        groupMrae = np.mean(groupDiff)

        allMrae.append(groupMrae)

    print('Worst group: ', np.max(allMrae))  # worst performing group
    print('Best group:  ', np.min(allMrae))  # best performing group
    print('Mean:        ', np.mean(allMrae))  # mean group performance

    return np.mean(allMrae)  # mean group performance


def weightedBackProjectionAccuracy(groundTruth, recovered, cameraResponse, labeledImage):
    """
    Compute the mean group performance of back projection accuracy. Spectra are grouped by the ``labelPixelGroup'' function
    :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
    :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
    :param cameraResponse: camera response functions. (Spectral_Dimension x RGB_Dimension)
    :param labeledImage: labeled image, output of ``labelPixelGroup'' function. (Height x Width)
    :return: mean group performance in MRAE
    """

    assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"
    assert groundTruth.shape[:2] == labeledImage.shape[:2], "Size not match for spectral and labeled images"
    assert groundTruth.shape[2] == cameraResponse.shape[0], "Spectral dimension mismatch between spectral images and camera response functions"

    specDim = cameraResponse.shape[0]  # spectral dimension

    # back projection + reshape the data into num_of_samples x spectral_dimensions
    groundTruthRGB = np.matmul(groundTruth.reshape(-1, specDim), cameraResponse)
    recoveredRGB = np.matmul(recovered.reshape(-1, specDim), cameraResponse)
    labelList = labeledImage.reshape(-1)

    # list of group numbers
    groups = np.sort(np.unique(labelList)).astype(int)

    allMrae = []  # used to collect mrae of all groups

    # group by group calculating mean MRAE
    for groupNum in groups:
        groupPixels = labelList == groupNum

        groupGroundTruth = groundTruthRGB[groupPixels, :]
        groupRecovered = recoveredRGB[groupPixels, :]

        # calculate MRAE
        groupDiff = np.abs(groupGroundTruth - groupRecovered) / groupGroundTruth
        groupMrae = np.mean(groupDiff)

        allMrae.append(groupMrae)

    print('Worst group: ', np.max(allMrae))  # worst performing group
    print('Best group:  ', np.min(allMrae))  # best performing group
    print('Mean:        ', np.mean(allMrae))  # mean group performance

    return np.mean(allMrae)  # mean group performance