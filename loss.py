import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class ContentLoss(nn.Module):
	def __init__(self, target):
		super(ContentLoss, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		self.loss = F.mse_loss(input, self.target)
		return input

class ContentLossChar(nn.Module):
	def __init__(self, target):
		super(ContentLossChar, self).__init__()
		self.target = target.detach()

	def forward(self, input):
		F2 = CharbonnierLoss()
		self.loss = F2(input, self.target)
		return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

    def gram_matrix(self, input):
    	a, b, c, d = input.size()  
    	features = input.view(a * b, c * d)  
        G = torch.mm(features, features.t())  
        return G.div(a * b * c * d)

class StyleLossChar(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        F2 = CharbonnierLoss()
        self.loss = F2(G, self.target)
        return input

    def gram_matrix(self, input):
    	a, b, c, d = input.size()  
    	features = input.view(a * b, c * d)  
        G = torch.mm(features, features.t())  
        return G.div(a * b * c * d)
