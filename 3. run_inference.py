#This script runs inference using the OpenVINO™ Inference Engine.# run_inference.py
from openvino.inference_engine import IECore
import numpy as np

def run_inference(ir_model_path, input_data):
    ie = IECore()
    net = ie.read_network(model=f'{ir_model_path}.xml', weights=f'{ir_model_path}.bin')
    exec_net = ie.load_network(network=net, device_name='CPU')
    
    input_blob = next(iter(net.input_info))
    output_blob = next(iter(net.outputs))
    
    result = exec_net.infer(inputs={input_blob: input_data})
    return result[output_blob]

if __name__ == "__main__":
    ir_model_path = 'ir_model/gpt2_model'
    input_data = np.random.randn(1, 10).astype(np.float32)  # Adjust input data as necessary
    
    result = run_inference(ir_model_path, input_data)
    print(f"Inference result: {result}")
