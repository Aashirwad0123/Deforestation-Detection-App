**Deforestation Detection App** 

A deep learning web application for detecting deforestation from satellite images using PyTorch and ResNet50. The app provides predictions, Grad-CAM visualizations, and an interactive interface for users to upload and analyze images.

**Features**

Classifies images as deforested or non-deforested
Grad-CAM heatmap visualization for model interpretability
User-friendly web interface (Streamlit)
Batch and single image prediction support
Model trained with regularization and hyperparameter tuning

**Flowchart**

![WhatsApp Image 2025-09-17 at 20 29 04_be22346c](https://github.com/user-attachments/assets/a6368ec8-22d1-44fa-a205-a76ced2297d3)

**Block Digram**

![WhatsApp Image 2025-09-17 at 20 27 32_d933f13c](https://github.com/user-attachments/assets/5e4c7e15-b601-4097-b2bf-4c1381229aa3)

**Model Performance **

![WhatsApp Image 2025-09-17 at 19 14 32_ac7be34c](https://github.com/user-attachments/assets/ea8773bd-a6b5-40b5-a5b2-d1cf708b1f32)


**Demo**

![WhatsApp Image 2025-09-17 at 20 32 04_00481306](https://github.com/user-attachments/assets/fbc3efa7-f455-40f2-87e2-71cd037aaef4)

![WhatsApp Image 2025-09-17 at 20 32 18_b43bf647](https://github.com/user-attachments/assets/f3e3e9ff-6cd0-4292-b43a-a312e3612da3)

![WhatsApp Image 2025-09-17 at 20 33 01_c15e5b88](https://github.com/user-attachments/assets/455fb64c-1a3b-4f21-9faa-a0cc899053d8)

![WhatsApp Image 2025-09-17 at 20 33 16_3500e8d7](https://github.com/user-attachments/assets/7488176e-e463-4462-8b54-e88a67cb6695)

**Getting Started**

Prerequisites

_Python 3.8+
pip_

Installation

**Clone the repository:**

git clone https://github.com/Aashirwad0123/Deforestation-Detection-App.git
cd Deforestation-Detection-App 

**Install dependencies:**

pip install -r requirements.txt

**Download the trained model:**

Place your best_model_1.pth file in the project root directory

**Running the App**

streamlit run app.py

Open the provided local URL in your browser.

**Usage**

Upload a satellite image (JPG, PNG, etc.)
View the prediction and Grad-CAM heatmap
Analyze results and download outputs if needed

**Project Structure**
.
├── app.py
├── main.ipynb
├── requirements.txt
├── best_model_1.pth
├── deforestation dataset/
│   ├── train data/
│   ├── val data/
│   └── test data/
└── ...

**Model Details**

Architecture: ResNet50 with dropout regularization

Framework: PyTorch, torchvision

Training: Data augmentation, weight decay, learning rate scheduling


**Acknowledgements**

PyTorch

Streamlit

Torchvision

**Project Members**

Aashirwad Mehare 

Ravindra Shelar 

Siddhesh K Mangarule

