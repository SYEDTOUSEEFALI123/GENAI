#This script uses the OpenVINO™ Model Optimizer to convert the ONNX model to Intermediate Representation (IR) format.#!/bin/bash

source /opt/intel/openvino/bin/setupvars.sh

mo --input_model gpt2_model.onnx --output_dir ir_model/
