from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from models import Model
from utils import ImageLoader

def func(content_path, style_path):
	imgloader = ImageLoader()
	content_img, style_image = imgloader.loadimages(content_path, style_path)
	cnn = models.vgg19(pretrained=True).features.cuda().eval()
	cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
	cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).cuda()
	input_img = content_img.clone()
	model = Model()
	output = model.run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_image, input_img)
	unloader = transforms.ToPILImage()
	image = output.cpu().clone()
	image = image.squeeze(0)
	image = unloader(image)
	image.save('./outputs/out1.jpg')

if __name__=="__main__":
	content_path = './content/chicago.jpg'
	style_path = './styles/candy.jpg'
	func(content_path, style_path)

