from PIL import Image
import os
import subprocess

# Function to resize image
def resize_img(path):
    try:
        im = Image.open(path)
        im = im.resize((768, 1024))
        im.save(path)
        print(f"Resized: {path}")
    except Exception as e:
        print(f"Error resizing {path}: {e}")

# Function to execute system command
def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

# Preprocessing images
try:
    cloth_dir = '/content/inputs/test/cloth/'
    for path in os.listdir(cloth_dir):
        resize_img(os.path.join(cloth_dir, path))

    # Remove .ipynb_checkpoints
    os.system("rm -rf /content/inputs/test/cloth/.ipynb_checkpoints")

    # Run cloth-mask.py
    os.chdir('/content/Upper-Body-Virtual-Try-On')
    run_command("python cloth-mask.py")

    # Back to the root directory
    os.chdir('/content')

    # Remove background
    run_command("python /content/Upper-Body-Virtual-Try-On/backgroundremoval.py")

    # Human parsing with Self-Correction-Human-Parsing
    run_command("python3 /content/Self-Correction-Human-Parsing/simple_extractor.py --dataset 'lip' --model-restore '/content/Self-Correction-Human-Parsing/checkpoints/final.pth' --input-dir '/content/inputs/test/image' --output-dir '/content/inputs/test/image-parse'")

    # OpenPose for pose estimation
    os.chdir('/content/openpose')
    run_command("./build/examples/openpose/openpose.bin --image_dir /content/inputs/test/image/ --write_json /content/inputs/test/openpose-json/ --display 0 --render_pose 0 --hand")
    run_command("./build/examples/openpose/openpose.bin --image_dir /content/inputs/test/image/ --display 0 --write_images /content/inputs/test/openpose-img/ --hand --render_pose 1 --disable_blending true")

    # Back to the root directory
    os.chdir('/content')

    # Create pairs text file
    model_images = os.listdir('/content/inputs/test/image')
    cloth_images = os.listdir('/content/inputs/test/cloth')
    pairs = zip(model_images, cloth_images)

    with open('/content/inputs/test_pairs.txt', 'w') as file:
        for model, cloth in pairs:
            file.write(f"{model} {cloth}")

    # Run virtual try-on model
    run_command("python /content/Upper-Body-Virtual-Try-On/test.py --name output --dataset_dir /content/inputs --checkpoint_dir /content/Upper-Body-Virtual-Try-On/checkpoints --save_dir /content/")

    # Clean up
    os.system("rm -rf /content/inputs")
    os.system("rm -rf /content/output/.ipynb_checkpoints")

except Exception as e:
    print(f"An error occurred: {e}")
