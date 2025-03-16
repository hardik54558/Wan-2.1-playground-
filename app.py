import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Replace with the actual model path on Hugging Face (e.g., "your-username/wan2.1-1.3B")
MODEL_NAME = "your-username/wan2.1-1.3B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to("cpu")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=5, placeholder="Enter your prompt..."),
    outputs="text",
    title="WAN 2.1 Playground",
    description="A simple Gradio interface for WAN 2.1"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
