# COE49413 Computer Vision Semester Project: License Plate Recognition and Angle Regression using YOLOv11 and CNN

This repository demonstrates a complete pipeline for **License Plate Recognition (LPR)** and **Angle Regression**. The project uses **YOLOv11** for detecting license plates and a custom **Convolutional Neural Network (CNN)** to estimate the angle of the detected plates for alignment.

## <a id="table-of-contents" />Table of Contents

  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Training the YOLOv11 Model](#training-the-yolov11-model)
  - [Training the Angle Regression CNN Model](#training-the-angle-regression-cnn-model)
  - [Testing the Trained Models](#testing-the-trained-models)
  - [Running the Complete Pipeline](#running-the-complete-pipeline)
  - [Code Breakdown](#code-breakdown)
    - [YOLOv11 Model Setup](#yolov11-model-setup)
    - [Angle Regression CNN Setup](#angle-regression-cnn-setup)
    - [Datasets](#datasets)
    - [Model Training and Validation](#model-training-and-validation)
    - [Inference and Testing](#inference-and-testing)

## <a id="prerequisites" />Prerequisites

Before running the code, ensure you have the following installed:

1. **Python** (>= 3.7)
2. **PyTorch** (>= 1.7)
3. **Numpy**
4. **OpenCV**
5. **Roboflow** for dataset download
6. **Matplotlib**
7. **Ultralytics YOLOv5**
8. **Onnxruntime**
9. **Pillow**

To install the necessary dependencies, run:

```python
!pip install opencv-python-headless numpy matplotlib ultralytics roboflow onnxruntime torch torchvision pillow
```

## <a id="setup" />Setup

1. **Clone the Repository:**

```bash
git clone https://github.com/Euova/license-plate-detection.git
```

## <a id="training-the-yolov11-model" />Training the YOLOv11 Model

### Step 1: Train YOLOv11 on License Plate Detection

The YOLOv11 model is pre-configured to detect license plates. To train the YOLOv11 model, use the following code:

```python
# Set device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Load pre-trained YOLOv11 model and move it to the device (GPU/CPU)
model = YOLO('yolo11n.pt').to(device)

# Start training YOLOv11
results = model.train(data="License-Plate-Recognition-4/data.yaml", epochs=50, device=device)
```

This will start training the model for 50 epochs. The dataset for training will be downloaded automatically using the Roboflow API key.

### Step 2: Model Evaluation

After training, you can evaluate the performance of your YOLO model on the validation set:

```python
# Evaluate the YOLOv11 model
results = model.val(device=device)
```

The evaluation results will provide metrics such as mAP (mean Average Precision) and other performance indicators.

### Step 3: Export Model to ONNX Format

Once training is completed, you can export the YOLOv11 model to an ONNX format for inference purposes:

```python
# Export the YOLOv11 model to ONNX format
model.export(format="onnx", device=device)
```

## <a id="training-the-angle-regression-cnn-model" />Training the Angle Regression CNN Model

### Step 1: Prepare the Dataset for Angle Prediction

The **Angle Regression CNN** is trained to predict the rotation angle of the license plate after detection. The dataset used consists of images with their respective normalized corner points labeled. We compute the angle in degrees using the corner points.

### Step 2: Train the Angle Regression CNN

```python
# Load the dataset
trainset = LicensePlateAngleDataset(image_dir="train/images", label_dir="train/labels", transform=train_transform)
valset = LicensePlateAngleDataset(image_dir="valid/images", label_dir="valid/labels", transform=transform)

# Set up DataLoader for training and validation
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

# Define the model, optimizer, and loss function
model = AngleRegressionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
criteria = nn.MSELoss()

# Start training the angle regression CNN
train_and_validate(model, trainloader, valloader, criteria, optimizer, epochs=5000)
```

### Step 3: Model Saving

After training is finished, save the model and model weights by using the following code:

```python
# Save the trained model
torch.save(model, 'final_angle_cnn_model.pth')
torch.save(model.state_dict(), 'final_angle_cnn_weights.pth')
```

## <a id="testing-the-trained-models" />Testing the Trained Models

After training both models, you can test their performance using a set of images.

### Testing YOLOv11 Model for License Plate Detection

To test the trained YOLOv11 model on new images, use the following code:

```python
# --- Load YOLOv11 ONNX Model ---
onnx_model_path = "runs/detect/train19/weights/best.onnx"
onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# --- Inference Function ---
def process_image(image_path):
    # Load and prepare image
    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    resized_img = cv.resize(img_rgb, (640, 640))
    img_input = resized_img.astype(np.float32).transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Run YOLOv11 ONNX model
    detections = onnx_session.run(None, {"images": img_input})[0]
    detections = process_detections(detections, 0.25)
    detections = apply_nms(detections)

    return detections
```

### Testing Angle Regression Model for Rotation Angle

The angle regression model is used to estimate the angle of the detected license plate. The detected plate is passed through the CNN to predict the angle.

```python
# --- Load the angle prediction model ---
angle_model = torch.load('best_angle_cnn_model.pth')
angle_model.eval()

# --- Preprocessing for angle model ---
angle_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

input_tensor = angle_transform(plate_img).unsqueeze(0).to(device)
        
with torch.no_grad():
    # Use the model to predict the angle of the license plate
    angle_deg = angle_model(input_tensor).item()
```

## <a id="running-the-complete-pipeline" />Running the Complete Pipeline

To run the complete pipeline, follow the steps below:

1. **Detect license plates using YOLOv11:** Detect license plates in images.
2. **Predict the angle:** Use the Angle Regression CNN to predict the orientation of the detected license plates.
3. **Display results:** Output the upright image of the plate with the predicted angle.


```python
# --- Load YOLOv11 ONNX Model ---
onnx_model_path = "runs/detect/train19/weights/best.onnx"
onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# --- Load the angle prediction model ---
angle_model = torch.load('best_angle_cnn_model.pth')
angle_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
angle_model.to(device)

# --- Preprocessing for angle model ---
angle_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Inference Function ---
def process_image(image_path):
    # Load and prepare image
    img = cv.imread(image_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    resized_img = cv.resize(img_rgb, (640, 640))
    img_input = resized_img.astype(np.float32).transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Show original image
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Run YOLOv11 ONNX model
    detections = onnx_session.run(None, {"images": img_input})[0]
    detections = process_detections(detections, 0.25)
    detections = apply_nms(detections)
    if len(detections) == 0:
        print(f"No detections in {image_path}")
        return None, None, None

    upright_plates = []
    angles = []
    corners_list = []

    for x, y, w, h, conf in detections:
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        x *= scale_x
        y *= scale_y
        w *= scale_x
        h *= scale_y

        x1, y1, x2, y2 = xywh_to_xyxy([x, y, w, h])
        crop = img_rgb[int(y1):int(y2), int(x1):int(x2)]

        if crop.size == 0:
            continue  # skip invalid crop

        plate_img = Image.fromarray(crop)
        input_tensor = angle_transform(plate_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            angle_deg = angle_model(input_tensor).item()
        
        corners = get_rotated_corners(x1, y1, x2, y2, angle_deg)
        sums = corners.sum(axis=1)
        top_left = corners[np.argmin(sums)]
        width = np.linalg.norm(corners[1] - corners[0])
        height = np.linalg.norm(corners[3] - corners[0])
        upright = get_upright_plate(img_rgb, top_left, width, height)

        upright_plates.append(upright)
        angles.append(angle_deg)
        corners_list.append(corners)

    return upright_plates, angles, corners_list
```

This will output the upright_plate (image), predicted angle, and corners of each detected license plate.

## <a id="code-breakdown" />Code Breakdown

### <a id="yolov11-model-setup" />YOLOv11 Model Setup
1. **YOLOv11 Setup:** The YOLOv11 model is used to detect license plates in the image. After training the model, it’s exported to the ONNX format for inference.

### <a id="angle-regression-cnn-setup" />Angle Regression CNN Setup
2. **Angle Prediction:** A CNN is used to predict the rotation angle of license plates, trained on annotated images with labeled corner coordinates.

### <a id="datasets" />Datasets

The two roboflow datasets used are the following:
1. **License Plate Recognition Dataset**: This dataset consists of images and their corresponding license plate bounding box corners. This dataset was used to train the YOLOv11 model to detect license plates. The dataset consists of: 

    - Train Images: 21,173
    - Validation Images: 2,046
    - Test Images: 1,020
    - Total Images: 24,239
2. **License Plates OBB**: This dataset consists of images and their corresponding license plate oriented bounding box corners. We converted the labels to be the angle of rotation of the license plate. This dataset was used to train the Angle Regresssion CNN model to predict the rotation of the detected license plate. The dataset consists of: 
    - Train Images: 979
    - Validation Images: 86
    - Test Images: 23
    - Total Images: 1,088

### <a id="model-training-and-validation" />Model Training and Validation
The training process involves training both the YOLOv11 model and the Angle Regression CNN using standard deep learning techniques such as backpropagation, optimization using Adam, and regularization techniques such as Data Augmentation and Dropout Layers.

### <a id="inference-and-testing" />Inference and Testing
Inferences are run on the trained models using new images, and results are displayed.
