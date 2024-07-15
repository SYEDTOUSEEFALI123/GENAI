#This script converts a PyTorch model to ONNX format.# convert_to_onnx.py
import torch
import torch.onnx
from transformers import GPT2Model

def convert_to_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'], opset_version=11)

if __name__ == "__main__":
    model = GPT2Model.from_pretrained('gpt2')
    dummy_input = torch.randn(1, 10)  # Adjust input size as necessary
    onnx_path = 'gpt2_model.onnx'
    
    convert_to_onnx(model, dummy_input, onnx_path)
    print(f"Model has been converted to ONNX and saved at {onnx_path}")
