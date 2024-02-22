import os
from PIL import Image
import numpy as np
from rembg import remove


class ImageProcessor:

    def __init__(self):
        self.original_width = None
        self.original_height = None
        self.original_image = None

        self.target_width = None
        self.target_height = None
        self.transformed_image = None
        self.save_path = None

    def remove_background(self, file_path: str):
        try:
            # Generate save path for the processed image
            self.save_path = file_path[:-3] + '.png'

            # Open the image and get its dimensions
            pic = Image.open(file_path)
            self.original_width = pic.width
            self.original_height = pic.height

            try:
                self.original_channels = np.asarray(pic).shape[2]
            except Exception as e:
                print("Single channel image and error", e)

            # Remove background and save the processed image
            os.remove(file_path)
            self.original_image = remove(pic)
            self.original_image.save(self.save_path)
            os.remove(self.save_path)

            return np.asarray(self.original_image)

        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            return None

    def transform_image(self, width=768, height=1024):
        try:
            new_size = (width, height)
            self.target_width = width
            self.target_height = height

            # Resize the image
            img = self.original_image.resize(new_size)
            self.transformed_image = img

            # Create a new RGBA image with a white background
            background = Image.new("RGBA", new_size, (255, 255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel

            # Save the transformed image as JPEG
            self.save_path = self.save_path[:-3] + '.jpg'
            background.convert('RGB').save(self.save_path, 'JPEG')

            return np.asarray(background.convert('RGB'))

        except Exception as e:
            print(f"Error transforming image: {e}")
            return None


# Usage of the class
image_processor = ImageProcessor()

# Process images in the specified directory
image_directory = '/content/inputs/test/image'
for image_name in os.listdir(image_directory):
    print(image_name)
    if image_name.endswith('jpg'):
        try:
            # Remove background
            processed_image = image_processor.remove_background(os.path.join(image_directory, image_name))

            if processed_image is not None:
                # Transform and save the image
                transformed_image = image_processor.transform_image(768, 1024)

                if transformed_image is not None:
                    # Further processing or saving logic can be added here
                    pass
                else:
                    print(f"Skipping image {image_name} due to transformation error.")

            else:
                print(f"Skipping image {image_name} due to background removal error.")

        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
