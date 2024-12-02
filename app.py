import os
import shutil
import tempfile
import gradio as gr
from PIL import Image
from rembg import remove
import subprocess
from glob import glob

def remove_background(input_url):
    # Create a temporary folder for downloaded and processed images
    temp_dir = tempfile.mkdtemp()

    # Download the image from the URL
    image_path = os.path.join(temp_dir, 'input_image.png')
    try:
        image = Image.open(requests.get(input_url, stream=True).raw)
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
    except Exception as e:
        shutil.rmtree(temp_dir)
        return f"Error removing background: {str(e)}"

    return removed_bg_path, temp_dir

def run_inference(temp_dir):
    # Define the inference configuration
    inference_config = "configs/inference-768-6view.yaml"
    pretrained_model = "pengHTYX/PSHuman_Unclip_768_6views"
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
    removed_bg_path, temp_dir = remove_background(input_url)

    if isinstance(removed_bg_path, str) and removed_bg_path.startswith("Error"):
        return removed_bg_path

    # Run inference
    output_images = run_inference(temp_dir)

    if isinstance(output_images, str) and output_images.startswith("Error"):
        shutil.rmtree(temp_dir)
        return output_images

    # Prepare outputs for display
    results = []
    for img_path in output_images:
        results.append((img_path, img_path))

    shutil.rmtree(temp_dir)  # Cleanup temporary folder
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
