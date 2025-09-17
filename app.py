import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Load model
@st.cache_resource
def load_model():
    from torchvision import models
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def grad_cam(model, image_tensor, target_class=None):
    gradients = []
    activations = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    def forward_hook(module, input, output):
        activations.append(output.detach())
    layer = model.layer4[-1].conv3
    handle_fw = layer.register_forward_hook(forward_hook)
    handle_bw = layer.register_backward_hook(backward_hook)
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    loss = output[0, target_class]
    model.zero_grad()
    loss.backward()
    grads = gradients[0]
    acts = activations[0]
    weights = grads.mean(dim=[2,3], keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    handle_fw.remove()
    handle_bw.remove()
    return cam

def main():
    st.title('Deforestation Detection App')
    st.write('Upload a satellite image to predict deforestation and visualize model attention (Grad-CAM).')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        transform = get_transform()
        image_tensor = transform(image).unsqueeze(0)
        model = load_model()
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
        classes = ['deforested', 'non-deforested']
        st.write(f'**Prediction:** {classes[pred]}')
        st.write(f'**Probabilities:** Deforested: {probs[0]:.2f}, Non-deforested: {probs[1]:.2f}')
        # Grad-CAM
        cam = grad_cam(model, image_tensor, target_class=pred)
        img_np = np.array(image.resize((224,224)))
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
        cmap = cv2.applyColorMap(np.uint8(255*cam_norm), cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(img_np, 0.6, cmap, 0.4, 0)
        st.image(overlay, caption=f'Grad-CAM: {classes[pred]}', use_column_width=True)

if __name__ == '__main__':
    main()
