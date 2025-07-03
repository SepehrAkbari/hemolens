import math
import os
import warnings
import mahotas
import numpy as np
import pandas as pd
import pywt
from skimage import color, exposure, filters, io, morphology, measure, transform, util, feature
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import gaussian, gabor, median
from skimage.morphology import disk

warnings.filterwarnings('ignore')

rs = 450

original_data_path = '../Data/bloodcells_dataset'
cleaned_data_path = '../Data/bloodcells_dataset_cleaned'

cell_types = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

def imNormalize(image):
    if len(image.shape) == 2:
        r = image
        g = image
        b = image
    else:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
    
    r = r - r.min()
    r = r / r.max()
    r = np.uint8(r * 255)
    
    g = g - g.min()
    g = g / g.max()
    g = np.uint8(g * 255)
    
    b = b - b.min()
    b = b / b.max()
    b = np.uint8(b * 255)
    
    return np.stack((r, g, b), axis=2)

def segment_cell(image, sigma=1, median_size=3, min_size=50, hole_area=50):
    gray_image = color.rgb2gray(image)
    
    smooth_image = gaussian(gray_image, sigma=sigma)
    smooth_image = median(smooth_image)
    
    thresh = filters.threshold_otsu(smooth_image)
    
    mask = smooth_image < thresh
    mask = morphology.remove_small_objects(mask, min_size=min_size)
    mask = morphology.remove_small_holes(mask, area_threshold=hole_area)
    
    labels = measure.label(mask)
    if labels.max() != 0:
        regions = measure.regionprops(labels)
        largest_region = max(regions, key=lambda r: r.area)
        mask = labels == largest_region.label

    segmented = image.copy()
    segmented[~mask] = 0

    return segmented, mask

def extract_color_histogram(image, num_bins=8):
    chans = []
    features = {}
    channels = ['R', 'G', 'B']
    for i in range(3):
        hist, _ = np.histogram(image[:,:,i], bins=num_bins, range=(0, 255))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        for b in range(num_bins):
            features[f"color_hist_{channels[i]}_bin{b}"] = hist[b]
    return features

def extract_hog_features(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    gray = color.rgb2gray(image)
    hog_vector, _ = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block, block_norm='L2-Hys',
                      visualize=True, feature_vector=True)
    features = {}
    for i, val in enumerate(hog_vector):
        features[f"hog_{i}"] = val
    return features

def extract_lbp_features(image, radius=1, n_points=8, num_bins=10):
    gray = color.rgb2gray(image)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    features = {}
    for i, val in enumerate(hist):
        features[f"lbp_bin{i}"] = val
    return features

def extract_gabor_features(image, frequency=0.6):
    gray = color.rgb2gray(image)
    filt_real, filt_imag = gabor(gray, frequency=frequency)
    features = {
        "gabor_mean": np.mean(filt_real),
        "gabor_std": np.std(filt_real)
    }
    return features

def extract_gist_features_with_keys(image, num_blocks=1, frequencies=[0.1, 0.3], thetas=[0, np.pi/2]):
    gray = color.rgb2gray(image)
    h, w = gray.shape
    block_h = math.floor(h / num_blocks)
    block_w = math.floor(w / num_blocks)
    features = {}
    
    for freq in frequencies:
        for theta in thetas:
            filt_real, _ = gabor(gray, frequency=freq, theta=theta)
            mean_val = np.mean(filt_real)
            std_val = np.std(filt_real)
            key_mean = f"gist_f{freq}_t{theta}_global_mean"
            key_std = f"gist_f{freq}_t{theta}_global_std"
            features[key_mean] = mean_val
            features[key_std] = std_val
    return features

def extract_hu_moments(mask):
    moments = measure.moments(mask.astype(float))
    hu = measure.moments_hu(moments)
    features = {}
    for i, val in enumerate(hu):
        features[f"hu_{i}"] = val
    return features

def extract_zernike_moments(mask, radius, degree_list=[2, 4]):
    features = {}
    mask_uint8 = (mask * 255).astype(np.uint8)
    center = (mask.shape[0] // 2, mask.shape[1] // 2)
    
    for degree in degree_list:
        zm_vector = mahotas.features.zernike_moments(mask_uint8, radius, degree, center)
        features[f"zernike_deg{degree}"] = np.mean(zm_vector)
    return features

def extract_wavelet_features(image):
    gray = color.rgb2gray(image)
    coeffs = pywt.dwt2(gray, 'haar')
    cA, (cH, cV, cD) = coeffs
    features = {
        "wavelet_mean": np.mean(cA),
        "wavelet_std": np.std(cA)
    }
    return features

def extract_haralick_features(image):
    gray = color.rgb2gray(image)
    gray_uint8 = (gray * 255).astype(np.uint8)
    glcm = feature.graycomatrix(gray_uint8, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        symmetric=True, normed=True)
    props = ['contrast', 'correlation', 'energy', 'homogeneity']
    features = {}
    for prop in props:
        feat = graycoprops(glcm, prop)
        features[f"haralick_{prop}"] = np.mean(feat)
    return features

def extract_features_dict(image):
    segmented, mask = segment_cell(image)
    
    features = {}

    features.update(extract_color_histogram(segmented, num_bins=8))
    features.update(extract_hog_features(segmented, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9))
    features.update(extract_lbp_features(segmented, radius=1, n_points=8, num_bins=10))
    features.update(extract_gabor_features(segmented, frequency=0.6))
    features.update(extract_gist_features_with_keys(segmented, num_blocks=1, frequencies=[0.1, 0.3], thetas=[0, np.pi/2]))
    features.update(extract_hu_moments(mask))
    features.update(extract_zernike_moments(mask, min(mask.shape) / 2, degree_list=[2,4]))
    features.update(extract_wavelet_features(segmented))
    features.update(extract_haralick_features(segmented))
    
    return features

data_records = []

for cell_type in cell_types:
    print(f"\nStarting {cell_type}")
    folder_path = os.path.join(cleaned_data_path, cell_type)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    count = 0
    for fname in image_files:
        print(f"Processing {count+1}/{len(image_files)}")
        count += 1
        file_path = os.path.join(folder_path, fname)
        try:
            image = io.imread(file_path)
            feat_dict = extract_features_dict(image)
            record = {
                'filename': file_path,
                'label': cell_type
            }
            record.update(feat_dict)
            data_records.append(record)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    print(f"{cell_type} done!\n")
    
print("\nSaving...")

df = pd.DataFrame(data_records)
df.to_csv('bloodcells_dataset_cleaned_features.csv', index=False)
print("Features saved")