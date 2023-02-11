from module.importer import *
from module.discriminator import *
from module.generator import *
from module.dataloader import *

dataset_path = './dataset/**/*.wav'

batch_size = 16

z_dim = 20

num_epochs = 500

lr = 0.0001

sampling_rate = 16000

D_updates_per_G_update = 5

generate_sounds_interval = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

train_sound_list = make_datapath_list(dataset_path)
train_dataset = GAN_Sound_Dataset(file_list=train_sound_list,device=device,batch_size=batch_size)

dataloader_for_G = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

dataloader_for_D = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)


def weights_init(m):
	if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d) or isinstance(m,nn.Linear):
		nn.init.kaiming_normal_(m.weight.data)


netG = Generator(z_dim=z_dim)

netG = netG.to(device)

netG.apply(weights_init)


netD = Discriminator()

netD = netD.to(device)

netD.apply(weights_init)

beta1 = 0.5
beta2 = 0.9
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,beta2))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,beta2))


G_losses = []
D_losses = []
iters = 0

generating_num = 5
z_sample = torch.Tensor(generating_num,z_dim).uniform_(-1,1).to(device)

print("Starting Training")

t_epoch_start = time.time()

for epoch in range(num_epochs):

	for generator_i,real_sound_for_G in enumerate(dataloader_for_G, 0):
		errD_loss_sum = 0
		for discriminator_i,real_sound_for_D in enumerate(dataloader_for_D, 0):
			if(discriminator_i==D_updates_per_G_update): break

			minibatch_size = real_sound_for_D.shape[0]

			if(minibatch_size==1): continue

			real_sound_for_D = real_sound_for_D.to(device)

			z = torch.Tensor(minibatch_size,z_dim).uniform_(-1,1).to(device)

			fake_sound = netG.forward(z)

			d_real = netD.forward(real_sound_for_D)

			d_fake = netD.forward(fake_sound)

			loss_real = d_real.mean()
			loss_fake = d_fake.mean()

			loss_gp = gradient_penalty(netD,real_sound_for_D.data,fake_sound.data,minibatch_size)
			beta_gp = 10.0

			errD = -loss_real + loss_fake + beta_gp*loss_gp

			optimizerD.zero_grad()

			errD.backward()

			optimizerD.step()

			errD_loss_sum += errD.item()
		
		
		minibatch_size = real_sound_for_G.shape[0]

		if(minibatch_size==1): continue

		real_sound_for_G = real_sound_for_G.to(device)

		z = torch.Tensor(minibatch_size,z_dim).uniform_(-1,1).to(device)

		fake_sound = netG.forward(z)

		d_fake = netD.forward(fake_sound)

		errG = -d_fake.mean()

		optimizerG.zero_grad()

		errG.backward()

		optimizerG.step()


		G_losses.append(errG.item())
		D_losses.append(errD_loss_sum/D_updates_per_G_update)

		iters += 1
	
	if (epoch%generate_sounds_interval==0 or epoch==num_epochs-1):
		print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
				% (epoch, num_epochs,
					errD_loss_sum/D_updates_per_G_update, errG.item()))

		output_dir = "./output/train/generated_epoch_{}".format(epoch)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		with torch.no_grad():
			generated_sound = netG(z_sample)
			save_sounds(output_dir,generated_sound,sampling_rate)


t_epoch_finish = time.time()
total_time = t_epoch_finish - t_epoch_start
with open('./output/train/time.txt', mode='w') as f:
	f.write("total_time: {:.4f} sec.\n".format(total_time))
	f.write("dataset size: {}\n".format(len(train_sound_list)))
	f.write("num_epochs: {}\n".format(num_epochs))
	f.write("batch_size: {}\n".format(batch_size))


torch.save(netG.to('cpu').state_dict(),"./output/generator_trained_model_cpu.pth")

plt.clf()
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('./output/train/loss.png')

print("data generated.")

