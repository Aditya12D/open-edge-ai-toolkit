import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import os

# Class mappings identical to the app
TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

WHEAT_CLASSES = [
    "healthy",
    "mildew",
    "yellow_rust",
    "brown_rust"
]

def format_label(label):
    if label.startswith("Tomato_"):
        label = label[7:]
    return label.replace("_", " ")

def predict(image_path, crop_type):
    if crop_type.lower() == 'tomato':
        model_path = 'assets/tomato.tflite'
        classes = TOMATO_CLASSES
    elif crop_type.lower() == 'wheat':
        model_path = 'assets/wheat.tflite'
        classes = WHEAT_CLASSES
    else:
        print("Invalid crop type. Choose 'tomato' or 'wheat'.")
        return

    # 1. Load TFLite model
    if not os.path.exists(model_path):
        print(f"Could not find model: {model_path}. Please run this script exactly from the app root directory.")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return
        
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Process image (Matches Flutter's Preprocessing exactly)
    try:
        # Load image, ensure RGB (in case of PNG with alpha), and resize to 224x224
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # Convert to float32 numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize: (image / 127.5) - 1.0 (Same as Flutter logic)
        img_array = (img_array / 127.5) - 1.0
        
        # Add batch dimension: Result is [1, 224, 224, 3]
        input_data = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Failed to load/process image {image_path}: {e}")
        return

    # 3. Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 4. Get output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]
    
    # 5. Extract results using argmax
    max_index = np.argmax(probabilities)
    max_prob = probabilities[max_index]
    predicted_label = classes[max_index]
    
    print(f"--- Results for {crop_type.capitalize()} ---")
    print(f"Raw Label Index : {max_index}")
    print(f"Raw Label String: {predicted_label}")
    print(f"Formatted UI    : {format_label(predicted_label)}")
    print(f"Confidence      : {max_prob * 100:.2f}%")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python test_predict.py <tomato|wheat> <path_to_image>")
        sys.exit(1)
        
    crop = sys.argv[1]
    img_path = sys.argv[2]
    predict(img_path, crop)
