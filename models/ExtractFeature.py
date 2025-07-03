import os
import numpy as np
import math
import warnings
import pickle
import mahotas
import pywt

warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self):
        pass

    def segment_cell(self, image, sigma=1, median_size=3, min_size=50, hole_area=50):
        from skimage import color, filters, morphology, measure
        from skimage.filters import gaussian, median

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
    
    def extract_color_histogram(self, image, num_bins=8):
        from skimage import color
        features = {}
        channels = ['R', 'G', 'B']
        for i in range(3):
            hist, _ = np.histogram(image[:,:,i], bins=num_bins, range=(0, 255))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-7)
            for b in range(num_bins):
                features[f"color_hist_{channels[i]}_bin{b}"] = hist[b]
        return features

    def extract_hog_features(self, image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
        from skimage import color
        from skimage.feature import hog
        gray = color.rgb2gray(image)
        hog_vector, _ = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                            cells_per_block=cells_per_block, block_norm='L2-Hys',
                            visualize=True, feature_vector=True)
        features = {}
        for i, val in enumerate(hog_vector):
            features[f"hog_{i}"] = val
        return features

    def extract_lbp_features(self, image, radius=1, n_points=8, num_bins=10):
        from skimage import color
        from skimage.feature import local_binary_pattern
        gray = color.rgb2gray(image)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=num_bins, range=(0, num_bins))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        features = {}
        for i, val in enumerate(hist):
            features[f"lbp_bin{i}"] = val
        return features

    def extract_gabor_features(self, image, frequency=0.6):
        from skimage import color
        from skimage.filters import gabor
        gray = color.rgb2gray(image)
        filt_real, filt_imag = gabor(gray, frequency=frequency)
        features = {
            "gabor_mean": np.mean(filt_real),
            "gabor_std": np.std(filt_real)
        }
        return features

    def extract_gist_features_with_keys(self, image, num_blocks=1, frequencies=[0.1, 0.3], thetas=[0, np.pi/2]):
        from skimage import color
        from skimage.filters import gabor
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

    def extract_hu_moments(self, mask):
        from skimage import measure
        moments = measure.moments(mask.astype(float))
        hu = measure.moments_hu(moments)
        features = {}
        for i, val in enumerate(hu):
            features[f"hu_{i}"] = val
        return features

    def extract_zernike_moments(self, mask, radius, degree_list=[2, 4]):
        features = {}
        mask_uint8 = (mask * 255).astype(np.uint8)
        center = (mask.shape[0] // 2, mask.shape[1] // 2)
        for degree in degree_list:
            zm_vector = mahotas.features.zernike_moments(mask_uint8, radius, degree, center)
            features[f"zernike_deg{degree}"] = np.mean(zm_vector)
        return features

    def extract_wavelet_features(self, image):
        from skimage import color
        gray = color.rgb2gray(image)
        coeffs = pywt.dwt2(gray, 'haar')
        cA, (cH, cV, cD) = coeffs
        features = {
            "wavelet_mean": np.mean(cA),
            "wavelet_std": np.std(cA)
        }
        return features

    def extract_haralick_features(self, image):
        from skimage import color, feature
        from skimage.feature import graycoprops
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

    def extract_features_dict(self, image):
        segmented, mask = self.segment_cell(image)
        features = {}
        features.update(self.extract_color_histogram(segmented, num_bins=8))
        features.update(self.extract_hog_features(segmented, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9))
        features.update(self.extract_lbp_features(segmented, radius=1, n_points=8, num_bins=10))
        features.update(self.extract_gabor_features(segmented, frequency=0.6))
        features.update(self.extract_gist_features_with_keys(segmented, num_blocks=1, frequencies=[0.1, 0.3], thetas=[0, np.pi/2]))
        features.update(self.extract_hu_moments(mask))
        radius_val = min(mask.shape) / 2
        features.update(self.extract_zernike_moments(mask, radius_val, degree_list=[2,4]))
        features.update(self.extract_wavelet_features(segmented))
        features.update(self.extract_haralick_features(segmented))
        return features

if __name__ == "__main__":
    extractor = FeatureExtractor()

    with open(os.path.join("feature_extractor.pkl"), "wb") as f:
        pickle.dump(extractor, f)