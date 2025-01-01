import numpy as np
import pymongo
from bson import Binary
import pickle
import os

class DBConnector:
    def __init__(self, source_folder):
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client.anomaly_detection
        self.collection = self.db.preprocessed_images
        self.source_folder = source_folder

    def save(self, images_info):

        # Save each image with a new name
        for image_info in images_info:
            # Create a filename (you can modify this as needed)
            full_path = os.path.join(self.source_folder, image_info['image_filename'])

            image_filename = full_path

            # Convert image to binary data
            with open(image_filename, "rb") as img_file:
                image_binary = img_file.read()

            image_info['image'] = image_binary

            # Insert the image data into MongoDB
            result = self.collection.insert_one({"image_data": image_info})

            # Print the inserted document ID
            print(f"Inserted document ID: {result.inserted_id}")

    def read(self, return_field='normalized_image_array_data'):
        documents = self.collection.find()
        data = [doc["image_data"]["normalized_image_array_data"] for doc in documents]

        return list(data)

