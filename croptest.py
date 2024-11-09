# Run this in a Streamlit app

# Import necessary libraries
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import os
import timm
from timm import create_model

# Set up device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# UI enhancements for a farmer-friendly interface
st.set_page_config(page_title="Crop Recognition and Health Status", page_icon="🌱", layout="wide")

st.title("🌾 Welcome to Farmers' Crop Assistant")
st.subheader("This tool helps you identify crop types, evaluate their health status, and predict potential diseases if the crop appears unhealthy. By simply uploading an image of your crop, our system can quickly analyze and provide insights to support better crop management and yield optimization.")

st.subheader("""
This portal provides three main features to assist you in crop monitoring:

1. **Crop Recognition**: Easily identify the type of crop by uploading an image. Our system leverages advanced machine learning models to recognize a wide variety of crops with high accuracy.
   
2. **Health Status Evaluation**: Check the health of your crop by uploading an image. The tool analyzes the crop image to determine if it’s healthy or showing signs of stress, such as nutrient deficiency or pest attacks.
   
3. **Disease Type Prediction**: If your crop is classified as unhealthy, this feature enables you to get a more detailed analysis. Upload an image of the affected crop, and our system will attempt to identify specific diseases, helping you take preventive or corrective measures.

### Benefits:
- **Fast and Accurate**: Our AI-powered tool provides quick analysis to help you make timely decisions.
- **Comprehensive Crop Coverage**: Works for a wide range of crop types to ensure flexibility.
- **Disease Prediction**: Get insights into potential diseases to take early action for better crop health.

**Upload your crop images and gain actionable insights in seconds to optimize your farming process!** 
""")


# Step-by-step Instructions
st.sidebar.title("How to Use This Tool")
st.sidebar.write("""
1. **Step 1: Crop Recognition**: Upload an image of your crop to identify its type.
2. **Step 2: Health Status Evaluation**: Upload an image to check your crop's health.
3. **Step 3: Disease Classification**: If the crop is unhealthy, upload an image to predict possible diseases.
""")

# Section 1: Crop Recognition
st.header("Step 1: Crop Recognition")
st.write("""
Upload an image of your crop to identify what it is. The tool uses machine learning to detect the crop type.
""")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained models for the ensemble
@st.cache_resource
def load_models():
    models = {
        'efficientnet': create_model('efficientnet_b0', pretrained=True).to(device),
        'seresnext': create_model('seresnext50_32x4d', pretrained=True).to(device),
        'deit': create_model('deit_small_patch16_224', pretrained=True).to(device),
        'mobilenetv3': create_model('mobilenetv3_large_100', pretrained=True).to(device),
        'resnet': create_model('resnet18', pretrained=True).to(device),
    }
    for name, model in models.items():
        model.head = torch.nn.Identity() if hasattr(model, 'head') else torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
    return models

models = load_models()

# Function to load and preprocess an image
def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image_transformed = transform(image).unsqueeze(0).to(device)
        return image_transformed
    except UnidentifiedImageError:
        st.error("Cannot identify the image file.")
        return None

# Function to extract features from a single image across all models
def extract_features_from_models(image_tensor):
    features = []
    with torch.no_grad():
        for model in models.values():
            feature = model(image_tensor).flatten().cpu().numpy()
            features.extend(feature)
    return np.array(features)

# Load dataset and split into train, validation, and test sets
@st.cache_data
def load_and_split_data(dataset_dir):
    images, labels = [], []
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                images.append(os.path.join(label_dir, image_file))
                labels.append(label)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder

# Extract features for a list of image paths
def extract_features_for_dataset(image_paths):
    all_features = []
    for image_path in tqdm(image_paths):
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)
        if image_tensor is not None:
            features = extract_features_from_models(image_tensor)
            all_features.append(features)
    return np.array(all_features)

# Train classifier
@st.cache_resource
def train_classifier(dataset_dir):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = load_and_split_data(dataset_dir)
    train_features = extract_features_for_dataset(X_train)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, y_train)
    return classifier, label_encoder, X_test, y_test

dataset_dir = st.text_input("Enter the dataset directory path:", "Padifier")
if dataset_dir:
    with st.spinner("Training classifier..."):
        classifier, label_encoder, X_test, y_test = train_classifier(dataset_dir)
    st.success("Classifier trained successfully!")

# Function to classify crop type and display the result
def classify_crop(image, classifier, label_encoder):
    image_tensor = preprocess_image(image)
    if image_tensor is not None:
        features = extract_features_from_models(image_tensor).reshape(1, -1)
        prediction = classifier.predict(features)
        crop_label = label_encoder.inverse_transform(prediction)[0]
         # Display the image with a defined width and height, and the result in large, bold text
        st.image(image, width=400, clamp=True)  # Set width to 400 pixels, adjust as needed
        st.markdown(f"<h2 style='text-align: center; color: white;'>Your crop is {crop_label}</h2>", unsafe_allow_html=True)
    else:
        st.error("Failed to process image for classification.")

# Upload and classify image
st.header("Upload crop image")
uploaded_file_crop = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="crop_uploader")
if uploaded_file_crop is not None:
    image = Image.open(uploaded_file_crop)
    classify_crop(image, classifier, label_encoder)

#STAGE 2

# Section 2: Health Status Evaluation
st.header("Step 2: Health Status Evaluation")
st.write("""
Now, let's check the health status of your crop. Upload an image and we'll determine whether your crop is healthy or not.
""")

# Run this code in a Streamlit app

# Set up device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained models for the ensemble
@st.cache_resource
def load_models():
    models = {
        'efficientnet': create_model('efficientnet_b0', pretrained=True).to(device),
        'seresnext': create_model('seresnext50_32x4d', pretrained=True).to(device),
        'deit': create_model('deit_small_patch16_224', pretrained=True).to(device),
        'mobilenetv3': create_model('mobilenetv3_large_100', pretrained=True).to(device),
        'resnet': create_model('resnet18', pretrained=True).to(device),
    }
    for name, model in models.items():
        model.head = torch.nn.Identity() if hasattr(model, 'head') else torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
    return models

models = load_models()

# Function to load and preprocess an image
def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image_transformed = transform(image).unsqueeze(0).to(device)
        return image_transformed
    except UnidentifiedImageError:
        st.error("Cannot identify the image file.")
        return None

# Function to extract features from a single image across all models
def extract_features_from_models(image_tensor):
    features = []
    with torch.no_grad():
        for model in models.values():
            feature = model(image_tensor).flatten().cpu().numpy()
            features.extend(feature)
    return np.array(features)

# Load dataset and split into train, validation, and test sets
@st.cache_data
def load_and_split_data(dataset_dir):
    images, labels = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for crop_type in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop_type)
        if os.path.isdir(crop_dir):
            for health_status in ["healthy", "unhealthy"]:
                health_dir = os.path.join(crop_dir, health_status)
                if os.path.isdir(health_dir):
                    for root, _, files in os.walk(health_dir):
                        for image_file in files:
                            if image_file.endswith(valid_extensions):
                                images.append(os.path.join(root, image_file))
                                labels.append(health_status)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder

# Extract features for a list of image paths
def extract_features_for_dataset(image_paths):
    all_features = []
    for image_path in tqdm(image_paths, desc="Extracting features", position=0, leave=False):
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)
        if image_tensor is not None:
            features = extract_features_from_models(image_tensor)
            all_features.append(features)
    return np.array(all_features)

# Train classifier
@st.cache_resource
def train_classifier(dataset_dir):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = load_and_split_data(dataset_dir)
    train_features = extract_features_for_dataset(X_train)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, y_train)
    return classifier, label_encoder, X_test, y_test

dataset_dir = st.text_input("Enter the dataset directory path:", "all_crops")
if dataset_dir:
    with st.spinner("Training classifier..."):
        classifier, label_encoder, X_test, y_test = train_classifier(dataset_dir)
    st.success("Classifier trained successfully!")

# Function to classify health status of a user-uploaded image
def classify_health_status(image, classifier, label_encoder):
    image_tensor = preprocess_image(image)
    if image_tensor is not None:
        features = extract_features_from_models(image_tensor).reshape(1, -1)
        prediction = classifier.predict(features)
        health_status = label_encoder.inverse_transform(prediction)[0]
       # Display the image with a defined width and the result in large, bold text
        st.image(image, width=400)  # Set the image width to 400 pixels
        st.markdown(f"<h2 style='text-align: center; color: white;'>Your crop is {health_status}</h2>", unsafe_allow_html=True)
    else:
        st.error("Failed to process image for health status evaluation.")

# Upload and classify image
st.header("Upload an Image for Health Status Evaluation")
uploaded_file_health = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="health_uploader")
if uploaded_file_health is not None:
    image = Image.open(uploaded_file_health)
    classify_health_status(image, classifier, label_encoder)

#STAGE 3

# Section 3: Disease Classification
st.header("Step 3: Disease Classification")
st.write("""
If your crop is unhealthy, upload an image to help us identify the disease affecting your crop.
""")

# Set up device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load pre-trained models for the ensemble
@st.cache_resource
def load_models():
    models = {
        'efficientnet': create_model('efficientnet_b0', pretrained=True).to(device),
        'seresnext': create_model('seresnext50_32x4d', pretrained=True).to(device),
        'deit': create_model('deit_small_patch16_224', pretrained=True).to(device),
        'mobilenetv3': create_model('mobilenetv3_large_100', pretrained=True).to(device),
        'resnet': create_model('resnet18', pretrained=True).to(device),
    }
    for name, model in models.items():
        model.head = torch.nn.Identity() if hasattr(model, 'head') else torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
    return models

models = load_models()

# Function to load and preprocess an image
def preprocess_image(image):
    try:
        image = image.convert('RGB')
        image_transformed = transform(image).unsqueeze(0).to(device)
        return image_transformed
    except UnidentifiedImageError:
        st.error("Cannot identify the image file.")
        return None

# Function to extract features from a single image across all models
def extract_features_from_models(image_tensor):
    features = []
    with torch.no_grad():
        for model in models.values():
            feature = model(image_tensor).flatten().cpu().numpy()
            features.extend(feature)
    return np.array(features)

# Load dataset for disease classification and split into train, validation, and test sets
def load_disease_data(dataset_dir):
    images, labels = [], []
    valid_extensions = ('.jpg', '.jpeg', '.png')

    for crop_type in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop_type, 'unhealthy')
        if os.path.isdir(crop_dir):
            for disease_type in os.listdir(crop_dir):
                disease_dir = os.path.join(crop_dir, disease_type)
                if os.path.isdir(disease_dir):
                    for image_file in os.listdir(disease_dir):
                        if image_file.endswith(valid_extensions):
                            images.append(os.path.join(disease_dir, image_file))
                            labels.append(disease_type)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder

# Extract features for a list of image paths
def extract_features_for_dataset(image_paths):
    all_features = []
    for image_path in tqdm(image_paths, desc="Extracting features", position=0, leave=False):
        image = Image.open(image_path)
        image_tensor = preprocess_image(image)
        if image_tensor is not None:
            features = extract_features_from_models(image_tensor)
            all_features.append(features)
    return np.array(all_features)

# Train classifier
@st.cache_resource
def train_classifier(dataset_dir):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = load_disease_data(dataset_dir)
    train_features = extract_features_for_dataset(X_train)
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(train_features, y_train)
    return classifier, label_encoder, X_test, y_test

# Allow user to input dataset path
dataset_dir = st.text_input("Enter the dataset directory path:", "all_crops", key="dataset_dir_input")
if dataset_dir:
    with st.spinner("Training disease classifier..."):
        classifier, label_encoder, X_test, y_test = train_classifier(dataset_dir)
    st.success("Disease classifier trained successfully!")

# Function to classify disease type for a user-uploaded "unhealthy" image
def classify_disease_type(image, classifier, label_encoder):
    image_tensor = preprocess_image(image)
    if image_tensor is not None:
        features = extract_features_from_models(image_tensor).reshape(1, -1)
        prediction = classifier.predict(features)
        disease_type = label_encoder.inverse_transform(prediction)[0]

        # Display the image and result in large, bold text
        st.image(image, width=400)
        st.markdown(f"<h2 style='text-align: center; color: white;'>Your crop has {disease_type}</h2>", unsafe_allow_html=True)
    else:
        st.error("Failed to process image for disease classification.")

# Upload and classify "unhealthy" crop image
st.header("Upload an Unhealthy Crop Image for Disease Classification")
uploaded_file_disease = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="disease_uploader")
if uploaded_file_disease is not None:
    image = Image.open(uploaded_file_disease)
    classify_disease_type(image, classifier, label_encoder)
