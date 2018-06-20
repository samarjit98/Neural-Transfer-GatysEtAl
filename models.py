from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from loss import StyleLoss, ContentLoss, StyleLossChar, ContentLossChar

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class Model:
	def __init__(self):
		self.content_layers = ['conv_4']
		self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

	def get_input_optimizer(self, input_img):
		optimizer = optim.LBFGS([input_img.requires_grad_()])
		return optimizer

	def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std, style_img, content_img):
	    cnn = copy.deepcopy(cnn)
	    normalization = Normalization(normalization_mean, normalization_std).cuda()
	    content_losses = []
	    style_losses = []
	    model = nn.Sequential(normalization)

	    i = 0  
	    for layer in cnn.children():
	        if isinstance(layer, nn.Conv2d):
	            i += 1
	            name = 'conv_{}'.format(i)
	        elif isinstance(layer, nn.ReLU):
	            name = 'relu_{}'.format(i)
	            layer = nn.ReLU(inplace=False)
	        elif isinstance(layer, nn.MaxPool2d):
	            name = 'pool_{}'.format(i)
	        elif isinstance(layer, nn.BatchNorm2d):
	            name = 'bn_{}'.format(i)
	        else:
	            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

	        model.add_module(name, layer)

	        if name in self.content_layers:
	            target = model(content_img).detach()
	            content_loss = ContentLoss(target)
	            model.add_module("content_loss_{}".format(i), content_loss)
	            content_losses.append(content_loss)

	        if name in self.style_layers:
	            target_feature = model(style_img).detach()
	            style_loss = StyleLoss(target_feature)
	            model.add_module("style_loss_{}".format(i), style_loss)
	            style_losses.append(style_loss)

	    for i in range(len(model) - 1, -1, -1):
	        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
	            break

	    model = model[:(i + 1)]
	    return model, style_losses, content_losses

	def run_style_transfer(self, cnn, normalization_mean, normalization_std, content_img, style_img, input_img, num_steps=300, style_weight=1000000, content_weight=1):
   
		print('Building the style transfer model..')
		model, style_losses, content_losses = self.get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img)
		optimizer = self.get_input_optimizer(input_img)

		print('Optimizing..')
		run = [0]
		while run[0] <= num_steps:

			def closure():
				input_img.data.clamp_(0, 1)
				optimizer.zero_grad()
				model(input_img)
				style_score = 0
				content_score = 0

				for sl in style_losses:
					style_score += sl.loss
				for cl in content_losses:
					content_score += cl.loss
				
				style_score *= style_weight
				content_score *= content_weight

				loss = style_score + content_score
				loss.backward()

				run[0] += 1
				if run[0] % 50 == 0:
					print("run {}:".format(run))
					print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.item(), content_score.item()))
					print()
				return style_score + content_score
			optimizer.step(closure)

		input_img.data.clamp_(0, 1)
		return input_img

