import os

import torch
import torchvision.transforms as transforms
from torchbench.datasets.utils import download_file_from_google_drive
from torchbench.image_classification import ImageNet

import rexnetv1

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Model 1
file_id = '1xeIJ3wb83uOowU008ykYj6wDX2dsncA9'
destination = './tmp/'
filename = 'rexnetv1_1.0x.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
sd = torch.load(os.path.join(destination, filename), map_location=torch.device('cpu'))
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = rexnetv1.ReXNetV1(width_mult=1.0)
model.load_state_dict(sd)
model.eval()

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ReXNetV1 1.0x',
    paper_arxiv_id='2007.00992',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.779, 'Top 5 Accuracy': 0.939},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()


# Model 2
file_id = '1x2ziK9Oyv66Y9NsxJxXsdjzpQF2uSJj0'
destination = './tmp/'
filename = 'rexnetv1_1.3x.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
sd = torch.load(os.path.join(destination, filename), map_location=torch.device('cpu'))
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = rexnetv1.ReXNetV1(width_mult=1.3)
model.load_state_dict(sd)
model.eval()

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ReXNetV1 1.3x',
    paper_arxiv_id='2007.00992',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.795, 'Top 5 Accuracy': 0.947},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()


# Model 3
file_id = '1TOBGsbDhTHWBgqcRnyKIR0tHsJTOPUIG'
destination = './tmp/'
filename = 'rexnetv1_1.5x.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
sd = torch.load(os.path.join(destination, filename), map_location=torch.device('cpu'))
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = rexnetv1.ReXNetV1(width_mult=1.5)
model.load_state_dict(sd)
model.eval()

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ReXNetV1 1.5x',
    paper_arxiv_id='2007.00992',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.803, 'Top 5 Accuracy': 0.952},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()


# Model 4
file_id = '1R1aOTKIe1Mvck86NanqcjWnlR8DY-Z4C'
destination = './tmp/'
filename = 'rexnetv1_2.0x.pth'
download_file_from_google_drive(file_id, destination, filename=filename)
sd = torch.load(os.path.join(destination, filename), map_location=torch.device('cpu'))
# Define the transforms need to convert ImageNet data to expected model input
input_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
model = rexnetv1.ReXNetV1(width_mult=2.0)
model.load_state_dict(sd)
model.eval()

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='ReXNetV1 2.0x',
    paper_arxiv_id='2007.00992',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    paper_results={'Top 1 Accuracy': 0.816, 'Top 5 Accuracy': 0.957},
    model_description="Official weights from the authors of the paper.",
)
torch.cuda.empty_cache()