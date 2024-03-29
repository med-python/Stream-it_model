import streamlit as st
import os
import zipfile
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from werkzeug.utils import secure_filename
import pydicom
import paramiko
import pandas as pd
import tempfile
import shutil

def download_model(remote_path, local_path, key_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('ec2-18-218-205-80.us-east-2.compute.amazonaws.com', username='ubuntu', key_filename=key_path)

    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()


# Загрузка предварительно обученной модели ResNet50
remote_path = '/home/ubuntu/service/models/resnet50_modelmammae.pth'
local_path = 'resnet50_modelmammae.pth'
key_path = "vkr.pem"
download_model(remote_path, local_path, key_path)

# Загрузка предварительно обученной модели ResNet50
resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50.to(device)

# Функция для классификации DICOM файла
def classify_dicom(filepath):
    # Функция нормализации и визуализации DICOM
    def normalize_visualize_dicom_1(dcm_file):
        dicom_file = pydicom.dcmread(dcm_file)
        dicom_array = dicom_file.pixel_array.astype(float)
        normalized_dicom_array = ((np.maximum(dicom_array, 0))/dicom_array.max()) * 255.0
        uint8_image = np.uint8(normalized_dicom_array)
        return uint8_image

    # Классификация DICOM файла
    image_1 = normalize_visualize_dicom_1(filepath)
    if image_1 is not None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img = Image.fromarray(image_1)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        resnet50.eval()
        with torch.no_grad():
            output = resnet50(img_tensor.to(device))
            predicted_class = torch.round(output).item()

        if predicted_class == 1:
            result_text = "Данное изображение соответствует 1 (или 2) категории по шкале BI-RADS_MRT."
        else:
            result_text = "Данное изображение соответствует 3 (или 4) категории по шкале BI-RADS_MRT. Требуется консультация специалиста."
        
        return result_text

def main():
    st.title('Классификация маммографических изображений')
    uploaded_file = st.sidebar.file_uploader("Загрузите zip-архив с файлами DICOM", type=['zip'])

    if uploaded_file is not None:
        # Создаем временную директорию для распаковки архива
        temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Получаем список всех файлов DICOM в директории
        dicom_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.dcm')]

        # Отображаем спиннер перед классификацией изображений
        with st.spinner('Классификация изображений...'):
            # Классифицируем каждое изображение
            for dicom_file in dicom_files:
                # Загружаем DICOM файл и получаем массив пикселей
                dicom_data = pydicom.dcmread(dicom_file)
                image_array = dicom_data.pixel_array

                # Классифицируем изображение и выводим результат классификации
                classification_result = classify_dicom(image_array)
                st.write("### Заключение классификатора:")
                st.write(classification_result)

        # Удаляем временную директорию после использования
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
