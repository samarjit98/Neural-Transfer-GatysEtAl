import torch
import torchvision
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

class ImageLoader:
	def __init__(self):
		self.imsize = 512 if torch.cuda.is_available() else 128
		self.imtransform = Compose([Resize(self.imsize), ToTensor()])

	def imageloader(self, imgname):
		img = Image.open(imgname)
		img = self.imtransform(img).unsqueeze(0)
		return img.cuda()

	def loadimages(self, content, style):
		return self.imageloader(content), self.imageloader(style