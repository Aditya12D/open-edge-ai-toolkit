import tensorflow as tf
import sys

def inspect_model(model_path):
    print(f"Inspecting {model_path}")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Inputs: {input_details[0]['shape']}")
        print(f"Outputs: {output_details[0]['shape']}")
    except Exception as e:
        print(f"Error: {e}")

inspect_model("c:/Users/Aditya Dev Sharma/Desktop/appforapp/flutter_application_1/assets/tomato.tflite")
inspect_model("c:/Users/Aditya Dev Sharma/Desktop/appforapp/flutter_application_1/assets/wheat.tflite")
