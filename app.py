import torch
import os
import shutil
import tempfile
import gradio as gr
from PIL import Image
from rembg import remove
import sys
import uuid
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

# Folder containing example images
examples_folder = "examples"

# Retrieve all file paths in the folder
images_examples = [
    os.path.join(examples_folder, file)
    for file in os.listdir(examples_folder)
    if os.path.isfile(os.path.join(examples_folder, file))
]

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
        unique_id = str(uuid.uuid4())
        removed_bg_path = os.path.join(temp_dir, f'output_image_rmbg_{unique_id}.png')
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

        
        # Retrieve the file name without the extension
        removed_bg_file_name = os.path.splitext(os.path.basename(removed_bg_path))[0]
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
    print(output_video)
    return output_video[0]

def gradio_interface():
    with gr.Blocks() as app:
        gr.Markdown("# PSHuman: Photorealistic Single-image 3D Human Reconstruction using Cross-Scale Multiview Diffusion and Explicit Remeshing")
        gr.HTML("""
        <div style="display:flex;column-gap:4px;">
            <a href="https://github.com/pengHTYX/PSHuman">
                <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
            </a> 
            <a href="https://penghtyx.github.io/PSHuman/">
                <img src='https://img.shields.io/badge/Project-Page-green'>
            </a>
			<a href="https://arxiv.org/pdf/2409.10141">
                <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
            </a>
            <a href="https://huggingface.co/spaces/fffiloni/PSHuman?duplicate=true">
				<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-sm.svg" alt="Duplicate this Space">
			</a>
			<a href="https://huggingface.co/fffiloni">
				<img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-me-on-HF-sm-dark.svg" alt="Follow me on HF">
			</a>
        </div>
        """)
        with gr.Row():
            
            with gr.Column(scale=2):
                input_image = gr.Image(
                    label="Image input", 
                    type="filepath",
                    height=240
                )
            
                submit_button = gr.Button("Process")
                gr.Examples(
                    examples = examples_folder,
                    inputs = [input_image],
                    examples_per_page = 4
                )

            output_video= gr.Video(label="Output Video", scale=3)

        submit_button.click(process_image, inputs=[input_image], outputs=[output_video])

    return app

# Launch the Gradio app
app = gradio_interface()
app.launch(show_api=False, show_error=True)
