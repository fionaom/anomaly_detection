from PIL import Image
import os
import numpy as np
from app.db_connector import DBConnector

class ImagePreProcessor:

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256
    IMAGE_CHANNELS = 3

    def __init__(self, folder_path, destination_folder, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
        self.folder_path = folder_path
        self.destination_folder = destination_folder
        self.target_size = target_size

    def load_images(self):
        image_files = [f for f in os.listdir(self.folder_path)]
        images_info = []
        for image_file in image_files:
            image_path = os.path.join(self.folder_path, image_file)
            image_filename = os.path.basename(image_path)
            img = Image.open(image_path)
            images_info.append({'image_filename': image_filename, 'image': img})
        return images_info

    def resize_images(self, images_info):
        for image_info in images_info:
            resized_image = image_info['image'].resize(self.target_size, Image.Resampling.LANCZOS)
            image_info['image'] = resized_image
        return images_info

    def normalize_images(self, images_info):
        for image_info in images_info:
            img_array = np.array(image_info['image']) / 255.0  # Scale pixels to [0, 1]
            array_as_list = img_array.tolist()
            image_info['normalized_image_array_data'] = array_as_list
        return images_info

    def save_images(self, images_info):
        # Create the save folder if it doesn't exist
        if not os.path.exists(self.destination_folder):
            os.makedirs(self.destination_folder)

        for idx, image_info in enumerate(images_info):
            full_path = os.path.join(self.destination_folder, image_info['image_filename'])
           # image_filename = os.path.basename(full_path)
            image_info['image'].save(full_path)
            print(f"Image saved as {full_path}")

    def preprocess_images(self):
        images_info = self.load_images()
        resized_images = self.resize_images(images_info)
        self.save_images(resized_images)
        self.normalize_images(resized_images)
        return images_info

if __name__ == "__main__":
    pre_processing = ImagePreProcessor(folder_path='../mvtech_datasets/metal_nut/train/good',
                                       destination_folder='./preprocessed_data/images')

    images_info = pre_processing.preprocess_images()

    # Save images to MongoDB
    mongoDBConnector = DBConnector(source_folder='./preprocessed_data/images')
    mongoDBConnector.save(images_info)
