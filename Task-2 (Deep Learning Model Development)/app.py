import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import torchvision

# Define model architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel()
model.load_state_dict(torch.load("cifar10_model.pt", map_location=device))
model.to(device)
model.eval()

# Class names
classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Image transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Streamlit layout
st.set_page_config(page_title="CIFAR-10 Image Classifier", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #6A0DAD; font-size: 48px; margin-bottom: 5px;'>CIFAR-10 Image Classifier</h1>
        <p style='font-size: 20px; color: #555;'>Upload an image and watch AI recognize what it sees!</p>
    </div>
    <hr style='border: 1px solid #ddd;'>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0] * 100
        predicted_class = torch.argmax(probs).item()

    st.markdown(f"""
        <div style='text-align: center;'>
            <h2 style='color: #4B0082;'>Prediction: <strong>{classes[predicted_class]}</strong></h2>
            <div style='height: 10px'></div>
            <div style='background-color: #eee; border-radius: 10px; height: 14px; width: 70%; margin: auto;'>
                <div style='width: {probs[predicted_class].item():.2f}%; background-color: #4169E1; height: 100%; border-radius: 10px;'></div>
            </div>
        </div>
        <div style='height: 30px'></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style='text-align: center;'>
            <h3 style='color: #333;'>Confidence for each class:</h3>
        </div>
        <ul style='font-size: 16px;'>
    """, unsafe_allow_html=True)

    for i, prob in enumerate(probs):
        st.markdown(f"<li>{classes[i]}: <strong>{prob.item():.2f}%</strong></li>", unsafe_allow_html=True)

    st.markdown("""
        </ul>
        <hr style='border: 1px solid #ccc;'>
        <div style='text-align: center;'>
            <p style='color: gray;'>This CIFAR-10 classifier was built using PyTorch and Streamlit.</p>
            <p style='color: #6A0DAD; font-weight: bold; font-size: 16px;'>Created by Abdul Rafay</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Please upload an image to get a prediction.")
