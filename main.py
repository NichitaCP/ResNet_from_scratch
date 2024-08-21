import streamlit as st
from ResNet import res_net_101
import os
import torch
from PIL import Image
from fn_utils import classify

st.title("Skin Cancer Classification")
st.header("Please upload an image of a mole.")
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

# Load model
net_101 = res_net_101(num_classes=7, block_input_layout=(32, 64, 128, 256))
net_101_path = os.path.join(os.getcwd(), "models", "net_101_v5.pth")
net101_state_dict = torch.load(net_101_path, weights_only=False, map_location=torch.device('cpu'))
net_101.load_state_dict(net101_state_dict)
net_101 = net_101.to("cpu")

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

sc_classes = [cls for cls in lesion_type_dict.items()]

if file is not None:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    results = classify(image, net_101, sc_classes)
    class_names = [result[0] for result in results]
    class_probs = [result[1] for result in results]

    st.write(f"# Here are the top 3 most likely diagnoses:")
    for class_name, class_prob in zip(class_names, class_probs):
        st.write(f"#### {class_name}: {class_prob*100:.2f}% chance.")
