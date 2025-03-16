import gradio as gr
import torch
# This is a placeholder import.
# Replace 'TextToVideoPipeline' with your actual pipeline from the library you use.
from diffusers import TextToVideoPipeline

# Replace with your model repository if you have uploaded your model.
# For testing, you might use a public model if available.
MODEL_NAME = "Hardik5456/WAN2.1-text-to-video"  # Change this if needed

# Load the text-to-video pipeline
pipe = TextToVideoPipeline.from_pretrained(MODEL_NAME)
pipe.to("cpu")  # Using CPU; expect slow results

def generate_video(prompt):
    result = pipe(prompt, num_frames=16)
    video = result.videos[0]
    return video

interface = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(lines=5, placeholder="Enter your text prompt here..."),
    outputs="video",
    title="WAN 2.1 Text-to-Video Playground",
    description="This is a demo for a 1.3B text-to-video model (CPU only, so it may be very slow)."
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
