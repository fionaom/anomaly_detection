import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

from app.db_connector import DBConnector
from app.image_pre_processor import ImagePreProcessor

# Visualize original and reconstructed images
import matplotlib.pyplot as plt
import pymongo

class AutoEncoderModelTrainer:
    def __init__(self, image_normalised_array_data):
        self.image_normalised_array = np.array(image_normalised_array_data)
        self.autoencoder = None

    def train(self):
        input_img = Input(shape=(ImagePreProcessor.IMAGE_WIDTH, ImagePreProcessor.IMAGE_HEIGHT, ImagePreProcessor.IMAGE_CHANNELS))  # Adjust shape to match your image dimensions

        # Encoder
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        # Autoencoder
        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        self.autoencoder.summary()

        x_train, x_test = train_test_split(self.image_normalised_array, test_size=0.2, random_state=42)

        history = self.autoencoder.fit(
            x_train, x_train,
            epochs=50,
            batch_size=32,
            shuffle=True,
            validation_data=(x_test, x_test)
        )

        self.autoencoder.save('./trained_model/autoencoder_model.keras')

        self.evaluate(x_test)

    # Evaluate on test set
    def evaluate(self, x_test):
        loss, accuracy = self.autoencoder.evaluate(x_test, x_test)
        print(f"Test Accuracy: {accuracy}")
        print(f"Reconstruction Loss: {loss}")

        # Predict reconstructed images
        reconstructed_images = self.autoencoder.predict(x_test[:10])


       # Denormalise images
        reconstructed_images = (reconstructed_images * 255).astype('uint8')
        original_images = (x_test * 255).astype('uint8')

        # Select a few images to visualize
        num_images = 5
        indices = range(num_images)

        plt.figure(figsize=(10, 4))
        for i, idx in enumerate(indices):
            # Original Image
            plt.subplot(2, num_images, i + 1)
            plt.imshow(x_test[idx], cmap='gray')
            plt.title("Original")
            plt.axis('off')

            # Reconstructed Image
            plt.subplot(2, num_images, i + 1 + num_images)
            plt.imshow(reconstructed_images[idx], cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Function to calculate reconstruction loss
    @staticmethod
    def calculate_reconstruction_loss(data, model):
        reconstructions = model.predict(data)
        reconstruction_errors = np.mean(np.abs(data - reconstructions), axis=1)
        return reconstruction_errors


if __name__ == "__main__":
    mongoDBConnector = DBConnector(source_folder='./preprocessed_data/images')
    image_normalised_array_data = mongoDBConnector.read()

    auto_encoder_model = AutoEncoderModelTrainer(image_normalised_array_data)
    auto_encoder_model.train()
