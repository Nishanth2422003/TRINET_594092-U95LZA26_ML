from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

sample_size = 16

z_dim = 20

sampling_rate = 16000

netG = Generator(z_dim=z_dim)
trained_model_path = "./output/generator_trained_model_cpu.pth"
netG.load_state_dict(torch.load(trained_model_path))

netG.eval()

noise = torch.Tensor(sample_size,z_dim).uniform_(-1,1)

generated_sound = netG(noise)

output_dir = "./output/inference"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

save_sounds("./output/inference/",generated_sound,sampling_rate)
