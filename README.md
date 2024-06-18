![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/273067258-7c1b9aee-b4e8-43b5-befd-588d4f0bd361.png)

# vision_task : detect and count product on shelves or in refrigerators

# Project Overview
This project leverages the YOLO v8 model for efficient product detection and counting. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. Our objective is to set up an environment where users can easily run the model using different formats (.pt, .onnx, and .trt) and interpret the results for various applications such as inventory management, automated checkout systems, and more.

# Validation
![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/validation/validation.png)
![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/validation/PR_curve.png)
![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/validation/F1_curve.png) 
![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/validation/P_curve.png)
![image](https://github.com/omarabdallah235/vision_task_shelves_count/blob/main/validation/R_curve.png)

# Setup and Installation
## Prerequisites
* Python 3.7 or later
* CUDA 10.2 or later (for GPU support)
* pip (Python package installer)

# Step 1: Create a Project Directory
Create a new directory for your project and navigate into it.

```
mkdir yolo-v8-project
cd yolo-v8-project
```
# Step 2: Create a Virtual Environment
Create and activate a virtual environment.

```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
# Step 3: Install Dependencies
Create a requirements.txt file in your project directory with the following contents:

```
ultralytics
numpy
opencv-python
torch
onnx
onnxruntime
pycuda

```
Then, install the dependencies:

```
pip install -r requirements.txt
```

# Step 4: Download the Model Files
Download the YOLO v8 model files in your desired formats (.pt, .onnx, and .trt) and place them in a directory named models within your project directory.
[download Here](https://drive.google.com/drive/folders/1YUcAVjZ3XgBaITGUw5GGTCZq8tATyRUE?usp=sharing)
# Step 5: Create the run_model.py Script
Create a file named run_model.py in your project directory and add the following code:

```python
import argparse
import cv2
import torch
import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit

def load_model(model_path):
    if model_path.endswith('.pt'):
        model = torch.load(model_path)
        return model
    elif model_path.endswith('.onnx'):
        ort_session = ort.InferenceSession(model_path)
        return ort_session
    elif model_path.endswith('.trt'):
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine
    else:
        raise ValueError("Unsupported model format")

def run_model(model, image_path, output_path):
    image = cv2.imread(image_path)
    if isinstance(model, torch.nn.Module):
        results = model(image)
    elif isinstance(model, ort.InferenceSession):
        ort_inputs = {model.get_inputs()[0].name: image}
        results = model.run(None, ort_inputs)
    elif isinstance(model, trt.ICudaEngine):
        # TensorRT inference code
        pass
    else:
        raise ValueError("Unsupported model type")

    cv2.imwrite(output_path, image)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO v8 model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image file")
    parser.add_argument("--output-path", type=str, default="output.jpg", help="Path to save the output image")
    args = parser.parse_args()

    model = load_model(args.model_path)
    run_model(model, args.image_path, args.output_path)
```


# Usage
## Running the Model
Using the .pt Model

```python
run_model.py --model-path models/yolov8.pt --image-path data/sample.jpg
```
Using the .onnx Model
```python
run_model.py --model-path models/yolov8.onnx --image-path data/sample.jpg
```
Using the .trt Model

```python
run_model.py --model-path models/yolov8.trt --image-path data/sample.jpg
```

# Product Detection and Counting
The script run_model.py accepts the following arguments:

* --model-path: Path to the model file.
* --image-path: Path to the image file to be processed.
* --output-path (optional): Path to save the output image with detections. Default is output.jpg.
  
## Example:


```python
run_model.py --model-path models/yolov8.pt --image-path data/sample.jpg --output-path results/output.jpg
```
# Results Interpretation
After running the model, the output will be an image with bounding boxes drawn around detected objects and a text file with the detection results.

# Bounding Boxes
Each detected object will have a bounding box with a label and confidence score. For example, a detected product might be labeled as "Product A: 0.95", where 0.95 is the confidence score.

# Text Output
A text file results/output.txt will be generated containing the details of the detections:

```yaml
Object Class: Product, Confidence: 0.95, Bounding Box: (x1, y1, x2, y2)
Object Class: Empty, Confidence: 0.87, Bounding Box: (x1, y1, x2, y2)
```









