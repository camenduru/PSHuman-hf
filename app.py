import torch
import os
import tarfile
import requests
import shutil
import tempfile
import gradio as gr
from PIL import Image
from rembg import remove
import sys
import subprocess
from glob import glob
import requests
from huggingface_hub import snapshot_download

# Define the URL and destination paths
onedrive_url = "https://hkustconnect-my.sharepoint.com/:u:/g/personal/plibp_connect_ust_hk/EZQphP-2y5BGhEIe8jb03i4BIcqiJ2mUW2JmGC5s0VKOdw?e=qVzBBD"
destination_tar = "smpl_related.tar.gz"
destination_folder = "smpl_related"

# Download the file
def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file to {destination}")
    else:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Extract the tar.gz file
def extract_tar(file_path, extract_to):
    print(f"Extracting {file_path} to {extract_to}...")
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extraction completed.")

# Ensure the folder exists
if not os.path.exists(destination_folder):
    try:
        # Step 1: Download the tar.gz file
        download_file(onedrive_url, destination_tar)

        # Step 2: Extract the tar.gz file
        extract_tar(destination_tar, "./")

        # Step 3: Clean up the tar.gz file after extraction
        os.remove(destination_tar)
        print(f"Cleaned up the tar file: {destination_tar}")

    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Folder {destination_folder} already exists. Skipping download and extraction.")

# Download models
os.makedirs("ckpts", exist_ok=True)

snapshot_download(
    repo_id = "pengHTYX/PSHuman_Unclip_768_6views",
    local_dir = "./ckpts"  
)


def remove_background(input_url):
    # Create a temporary folder for downloaded and processed images
    temp_dir = tempfile.mkdtemp()

    # Download the image from the URL
    image_path = os.path.join(temp_dir, 'input_image.png')
    try:
        image = Image.open(input_url).convert("RGBA")
        image.save(image_path)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error downloading or saving the image: {str(e)}"

    """
    # Run background removal
    try:
        removed_bg_path = os.path.join(temp_dir, 'output_image_rmbg.png')
        img = Image.open(image_path)
        result = remove(img)
        result.save(removed_bg_path)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error removing background: {str(e)}"

    return removed_bg_path, temp_dir
    """
    return image_path, temp_dir
    
def run_inference(temp_dir):
    # Define the inference configuration
    inference_config = "configs/inference-768-6view.yaml"
    pretrained_model = "./ckpts"
    crop_size = 740
    seed = 600
    num_views = 7
    save_mode = "rgb"

    try:
        # Run the inference command
        subprocess.run(
            [
                "python", "inference.py",
                "--config", inference_config,
                f"pretrained_model_name_or_path={pretrained_model}",
                f"validation_dataset.crop_size={crop_size}",
                f"with_smpl=false",
                f"validation_dataset.root_dir={temp_dir}",
                f"seed={seed}",
                f"num_views={num_views}",
                f"save_mode={save_mode}"
            ],
            check=True
        )

        # Collect the output images
        output_images = glob(os.path.join(temp_dir, "*.png"))
        return output_images
    except subprocess.CalledProcessError as e:
        return f"Error during inference: {str(e)}"

def process_image(input_url):
    # Remove background
    result = remove_background(input_url)
    
    if isinstance(result, str) and result.startswith("Error"):
        raise gr.Error(f"{result}")  # Return the error message if something went wrong

    removed_bg_path, temp_dir = result  # Unpack only if successful

    # Run inference
    output_images = run_inference(temp_dir)

    if isinstance(output_images, str) and output_images.startswith("Error"):
        shutil.rmtree(temp_dir)
        raise gr.Error(f"{output_images}")   # Return the error message if inference failed

    # Prepare outputs for display
    results = []
    for img_path in output_images:
        results.append((img_path, img_path))

    #shutil.rmtree(temp_dir)  # Cleanup temporary folder
    return results

def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Background Removal and Inference Pipeline")

        with gr.Row():
            input_image = gr.Image(label="Image input", type="filepath")
            submit_button = gr.Button("Process")

        output_gallery = gr.Gallery(label="Output Images")

        submit_button.click(process_image, inputs=[input_image], outputs=[output_gallery])

    return app

# Launch the Gradio app
app = gradio_interface()
app.launch()
