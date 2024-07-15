#Step 4: Inference with OpenVINO

from openvino.inference_engine import IECore

# Initialize Inference Engine
ie = IECore()
net = ie.read_network(model='openvino_model/emotion_recognition_model.xml', weights='openvino_model/emotion_recognition_model.bin')
exec_net = ie.load_network(network=net, device_name='CPU')

# Prepare input
input_blob = next(iter(net.input_info))
output_blob = next(iter(net.outputs))

# Perform inference
def predict_expression(image):
    p_image = cv2.resize(image, (48, 48))
    p_image = p_image / 255.0
    p_image = np.expand_dims(p_image, axis=0)
    p_image = np.expand_dims(p_image, axis=-1)

    result = exec_net.infer(inputs={input_blob: p_image})
    return np.argmax(result[output_blob])

# Test inference
test_image = x_test[0]
expression = predict_expression(test_image)
print(f'Predicted expression: {expression}')
