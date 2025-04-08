from PIL import Image
import os


def resize_and_convert_images(input_folder, output_folder, target_size=(1024, 1024)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    image_files.sort()

    for i, filename in enumerate(image_files, start=1):
        image_path = os.path.join(input_folder, filename)
        try:
            image = Image.open(image_path)
            resized_image = image.resize(target_size)
            new_filename = f"{str(i).zfill(3)}A.png"
            output_path = os.path.join(output_folder, new_filename)
            resized_image.save(output_path, 'PNG')
            print(f"Successfully resized and converted {filename} to {new_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    input_folder = 'car15'
    output_folder = 'car16'
    resize_and_convert_images(input_folder, output_folder)
