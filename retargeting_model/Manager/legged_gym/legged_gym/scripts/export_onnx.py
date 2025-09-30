"""
This file is used to transfer a .pt file to a .onnx file
"""
import torch
import os

def export_jit_to_onnx(jit_model, path, dummy_input):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # export jit to onnx
    torch.onnx.export(
        jit_model,                  
        dummy_input,            
        path,                       
        export_params=True,         
        opset_version=11,           
        do_constant_folding=True,   
        input_names=['input'],      
        output_names=['output'],    
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported JIT model to ONNX at: {path}")

pt_path = ""
jit_model = torch.jit.load(pt_path) # path to your .pt file
dummy_input = torch.randn(1, 76*6, device='cpu')  # shape of the input of the model
export_path = ""

export_jit_to_onnx(jit_model, export_path, dummy_input)