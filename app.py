import json
import os

import torch
import torchvision

import model

import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

cuda = True if torch.cuda.is_available() else False
device = torch.device('cpu')
if cuda:
    device = torch.device('cuda')

app = Flask(__name__)
saved_weights = "savedWeights/87.42%_dataset.npy"
model = model.ResNet50().to(device)
model.load_state_dict(torch.load(saved_weights))
model.eval()

img_class_map = None

mapping_file_path = 'birds.json'
if os.path.isfile(mapping_file_path):
    with open(mapping_file_path) as f:
        img_class_map = json.load(f)


def transform_image(infile):
    input_transforms = [
    transforms.Resize((70, 70)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]
    my_transforms = torchvision.transforms.Compose(input_transforms)
    image = Image.open(infile)
    timg = my_transforms(image)
    timg.unsqueeze_(0)
    return timg


def get_prediction(input_tensor):
    input_tensor = input_tensor.cuda()
    outputs = model.forward(input_tensor)
    _, y_hat = outputs.max(1)
    prediction = y_hat.item()
    return prediction


def render_prediction(prediction_idx):
    stridx = str(prediction_idx)
    class_name = 'Unknown'
    if img_class_map is not None:
        if stridx in img_class_map is not None:
            class_name = img_class_map[stridx][1]
    return prediction_idx, class_name


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/test', methods=['GET'])
def root():
    return jsonify({'msg': 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is not None:
            input_tensor = transform_image(file)
            prediction_idx = get_prediction(input_tensor)
            class_id, class_name = render_prediction(prediction_idx)
            return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()
