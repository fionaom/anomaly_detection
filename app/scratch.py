

from PIL import Image
import numpy as np

# Load an image
image = Image.open('../mvtech_datasets/metal_nut/train/good/000.png')  # Replace 'image.jpg' with your image file

# Convert the image to a NumPy array
image_array = np.array(image)

# Print the 3D array
print(image_array)


# Optional: Display some details about the array
print(f"Shape of the array: {image_array.shape}") # e.g., (height, width, 3) for RGB images
print(f"Data type: {image_array.dtype}")

# Iterate through the pixels
height, width, channels = image_array.shape

for y in range(height):
    for x in range(width):
        rgb_values = image_array[y, x]  # Access RGB values at (x, y)
        print(f"x: {x}, y: {y}, RGB: {rgb_values}")

# Convert to grayscale (if needed)
gray_image = image.convert('L')

# Access pixel values
pixels = np.array(gray_image)  # Convert to NumPy array
print(pixels)
pixel_value = pixels[50, 50]   # Get the pixel value at row 50, column 50
print("Pixel value:", pixel_value)


# Print the shape of the array
print("Pixel array shape:", pixels.shape)


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Convert the image to a NumPy array
image_array = np.array(image)

# If the image has an alpha channel, remove it for simplicity
if image_array.shape[-1] == 4:
    image_array = image_array[:, :, :3]

# Convert RGB to grayscale for colormap application (optional)
grayscale_image = np.mean(image_array, axis=2)

# Display the image with the Viridis colormap
plt.imshow(grayscale_image, cmap='viridis')
plt.colorbar()  # Optional: Add a colorbar to show the mapping
plt.title("Image in Viridis Colormap")
plt.show()
