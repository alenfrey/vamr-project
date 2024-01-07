import cv2

# Default configurations
default_config = {
    "matcher": cv2.BFMatcher(),
    "detector": cv2.SIFT_create(),
    "lowe_ratio": 0.7,
    "match_max_dist": 50,
    "ransac": {"prob": 0.999, "threshold": 1.0},
}

# Specific configurations for datasets
config = {
    "parking": {
        "initialization": {
            "detector": cv2.SIFT_create(
                nfeatures=2000,
                nOctaveLayers=5,
                contrastThreshold=0.05,
                edgeThreshold=15,
                sigma=1.6,
            ),
            "matcher": cv2.BFMatcher(),
            "lowe_ratio": 0.7,
            "match_max_dist": 30,
            "ransac": {"prob": 0.999, "threshold": 2.0},
            "translation_scale": 0.3,
        },
        "continuous": {
            "detector": cv2.SIFT_create(
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=60,
                sigma=1.6,
            ),
            "matcher": cv2.FlannBasedMatcher(),
            "lowe_ratio": 0.5,
            "match_max_dist": 50,
            "ransac": {"prob": 0.999, "threshold": 1.0},
            "translation_scale": 0.15,
        },
    },
    "kitti": {
        "initialization": {
            "detector": cv2.SIFT_create(
                nfeatures=2000,
                nOctaveLayers=5,
                contrastThreshold=0.05,
                edgeThreshold=15,
                sigma=1.6,
            ),
            "matcher": cv2.BFMatcher(),
            "lowe_ratio": 0.7,
            "match_max_dist": 30,
            "ransac": {"prob": 0.999, "threshold": 2.0},
            "translation_scale": 0.5,
        },
        "continuous": {
            "detector": cv2.SIFT_create(
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=60,
                sigma=1.6,
            ),
            "matcher": cv2.FlannBasedMatcher(),
            "lowe_ratio": 0.8,
            "match_max_dist": 50,
            "ransac": {"prob": 0.999, "threshold": 1.0},
            "translation_scale": 0.5,
        },
    },
    "malaga-urban-dataset-extract-07": {
        "initialization": {
            "detector": cv2.SIFT_create(
                nfeatures=2000,
                nOctaveLayers=5,
                contrastThreshold=0.05,
                edgeThreshold=15,
                sigma=1.6,
            ),
            "matcher": cv2.BFMatcher(),
            "lowe_ratio": 0.7,
            "match_max_dist": 30,
            "ransac": {"prob": 0.999, "threshold": 2.0},
            "translation_scale": 0.5,
        },
        "continuous": {
            "detector": cv2.SIFT_create(
                nOctaveLayers=3,
                contrastThreshold=0.02,
                edgeThreshold=60,
                sigma=1.6,
            ),
            "matcher": cv2.FlannBasedMatcher(),
            "lowe_ratio": 0.5,
            "match_max_dist": 50,
            "ransac": {"prob": 0.999, "threshold": 1.0},
            "translation_scale": 0.5,
        },
    },
}


def get_config(dataset_name, mode):
    dataset_config = config.get(dataset_name, {})
    common_config = dataset_config.get("common", {})
    mode_config = dataset_config.get(mode, {})

    # merge configurations: default < common < mode-specific
    final_config = {**default_config, **common_config, **mode_config}
    return final_config
