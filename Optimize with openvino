#Step 3: Optimize and Deploy with OpenVINO


# Save the model in TensorFlow SavedModel format
model.save('emotion_recognition_model')

# Convert the model to OpenVINO IR format
!mo --saved_model_dir emotion_recognition_model --output_dir openvino_model

# Use OpenVINO Inference Engine for inference
