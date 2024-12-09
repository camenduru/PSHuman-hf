import torch
import os
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

# Download models
os.makedirs("ckpts", exist_ok=True)

snapshot_download(
    repo_id = "pengHTYX/PSHuman_Unclip_768_6views",
    local_dir = "./ckpts"  
)

os.makedirs("smpl_related", exist_ok=True)
snapshot_download(
    repo_id = "fffiloni/PSHuman-SMPL-related",
    local_dir = "./smpl_related"  
)


def remove_background(input_url):
    # Create a temporary folder for downloaded and processed images
    temp_dir = tempfile.mkdtemp()

    # Download the image from the URL
    image_path = os.path.join(temp_dir, 'input_image.png')
    try:
        image = Image.open(input_url)
        image.save(image_path)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error downloading or saving the image: {str(e)}"

    # Run background removal
    try:
        removed_bg_path = os.path.join(temp_dir, 'output_image_rmbg.png')
        img = Image.open(image_path)
        result = remove(img)
        result.save(removed_bg_path)

        # Remove the input image to keep the temp directory clean
        os.remove(image_path)
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error removing background: {str(e)}"

    return removed_bg_path, temp_dir
    
def run_inference(temp_dir, removed_bg_path):
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

        
        # Retrieve the file name
        removed_bg_file_name = os.path.basename(removed_bg_path)
        output_videos = glob(os.path.join(f"out/{removed_bg_file_name}", "*.mp4"))
        return output_videos
    except subprocess.CalledProcessError as e:
        return f"Error during inference: {str(e)}"

def process_image(input_url):
    # Remove background
    result = remove_background(input_url)
    
    if isinstance(result, str) and result.startswith("Error"):
        raise gr.Error(f"{result}")  # Return the error message if something went wrong

    removed_bg_path, temp_dir = result  # Unpack only if successful

    # Run inference
    output_video = run_inference(temp_dir, removed_bg_path)

    if isinstance(output_video, str) and output_video.startswith("Error"):
        shutil.rmtree(temp_dir)
        raise gr.Error(f"{output_images}")   # Return the error message if inference failed

    
    shutil.rmtree(temp_dir)  # Cleanup temporary folder
    return output_video[0]

def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# Background Removal and Inference Pipeline")

        with gr.Row():
            input_image = gr.Image(label="Image input", type="filepath")
            submit_button = gr.Button("Process")

        output_video= gr.Video(label="Output Video")

        submit_button.click(process_image, inputs=[input_image], outputs=[output_video])

    return app

# Launch the Gradio app
app = gradio_interface()
app.launch()
