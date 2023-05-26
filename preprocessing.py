import pandas as pd
import cv2
from sklearn.model_selection import train_test_split


def load_csv(path: str = "dataset/anno_train.csv", nrows: int = None):
    """Read csv file"""
    data = pd.read_csv(path, nrows=nrows)
    return data


def __preprocess_image(image_path):
    image = cv2.imread(f"car_ims/cars_train/{image_path}")
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values
    return image


def prepare(data):
    """Prepare the data to train\n
    :return data"""
    data['image'] = data['relative_im_path'].apply(__preprocess_image)
    # print(len(data.values[:, -1]))
    img_width, img_height, num_channels = 224, 224, 4
    data = data.reshape(data.shape[0], img_width, img_height, num_channels)

    return data


def extract_target(data):
    """Extract targets from data"""
    targets = data[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values

    return targets


def split_train_test(data, targets):
    train_data, test_data, train_targets, test_targets = train_test_split(
        data['image'].values, targets, test_size=0.2)
    return train_data, test_data, train_targets, test_targets
