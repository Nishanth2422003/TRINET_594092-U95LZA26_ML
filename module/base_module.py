from .importer import *

class PhaseShuffle(nn.Module):

	def __init__(self,n):
		super().__init__()
		self.n = n

	def forward(self, x):

		if self.n == 0:
			return x

		shift = torch.Tensor(x.shape[0]).random_(-self.n,self.n+1).type(torch.int)

		x_shuffled = x.clone()
		for i,shift_num in enumerate(shift):
			if(shift_num==0): continue
			dim = len(x_shuffled[i].size()) - 1
			origin_length = x[i].shape[dim]
			if shift_num > 0:
				left = torch.flip(torch.narrow(x[i],dim,1,shift_num),[dim])
				right = torch.narrow(x[i],dim,0,origin_length-shift_num)
			else:
				shift_num = -shift_num
				left = torch.narrow(x[i],dim,shift_num,origin_length-shift_num)
				right = torch.flip(torch.narrow(x[i],dim,origin_length-shift_num-1,shift_num),[dim])
			x_shuffled[i] = torch.cat([left,right],dim)

		return x_shuffled

def gradient_penalty(netD,real,fake,batch_size,gamma=1):
	device = real.device

	alpha = torch.rand(batch_size,1,1,requires_grad=True).to(device)

	x = alpha*real + (1-alpha)*fake

	d_ = netD.forward(x)

	g = torch.autograd.grad(outputs=d_, inputs=x,
							grad_outputs=torch.ones(d_.shape).to(device),
							create_graph=True, retain_graph=True,only_inputs=True)[0]
	g = g.reshape(batch_size, -1)
	return ((g.norm(2,dim=1)/gamma-1.0)**2).mean()


