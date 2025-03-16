from diffusers import TextToVideoSDPipeline
import torch
import gradio as gr

# Replace with your actual model repository if available.
# For testing, you may need to use a public model checkpoint.
MODEL_NAME = "Hardik5456/WAN2.1-text-to-video"  

# Load the text-to-video pipeline.
# Using torch_dtype=torch.float16 for reduced memory usage.
pipe = TextToVideoSDPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
# Set the pipeline to run on CPU (this will be very slow)
pipe.to("cpu")

def generate_video(prompt):
    # Generate a video using the prompt.
    # You can adjust num_frames, num_inference_steps, and guidance_scale as needed.
    result = pipe(prompt, num_frames=16, num_inference_steps=50, guidance_scale=9.0)
    # Return the first video's frames (as a video output in Gradio)
    return result.frames[0]

# Create a Gradio interface.
demo = gr.Interface(
    fn=generate_video,
    inputs=gr.Textbox(lines=5, placeholder="Enter text prompt..."),
    outputs="video",
    title="WAN 2.1 Text-to-Video Playground",
    description="Prototype for text-to-video generation using WAN 2.1 1.3B model (CPU only; very slow)."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
