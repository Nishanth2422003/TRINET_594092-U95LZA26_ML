from .importer import *

def make_datapath_list(target_path):
	path_list = []
	for path in glob.glob(target_path,recursive=True):
		path_list.append(path)

	print("sounds : " + str(len(path_list)))
	return path_list

class GAN_Sound_Dataset(data.Dataset):

	def __init__(self,file_list,device,batch_size,sound_length=65536,sampling_rate=16000,dat_threshold=1100):

		self.file_list = file_list
		self.device = device
		self.batch_size = batch_size
		self.sound_length = sound_length
		self.sampling_rate = sampling_rate
		self.dat_threshold = dat_threshold

		if(len(self.file_list)<=dat_threshold):
			self.file_contents = []
			for file_path in self.file_list:

				sound,_ = librosa.load(file_path,sr=self.sampling_rate)
				self.file_contents.append(sound)

	def __len__(self):
		return max(self.batch_size, len(self.file_list))

	def __getitem__(self,index):
		if(len(self.file_list)<=self.dat_threshold):
			sound = self.file_contents[index%len(self.file_list)]
		else:
			sound_path = self.file_list[index%len(self.file_list)]

			sound,_ = librosa.load(sound_path,sr=self.sampling_rate)

		sound = (torch.from_numpy(sound.astype(np.float32)).clone()).to(self.device)

		max_amplitude = torch.max(torch.abs(sound))
		if max_amplitude > 1:
			sound /= max_amplitude

		loaded_sound_length = sound.shape[0]

		if loaded_sound_length < self.sound_length:
			padding_length = self.sound_length - loaded_sound_length
			left_zeros = torch.zeros(padding_length//2).to(self.device)
			right_zeros = torch.zeros(padding_length - padding_length//2).to(self.device)
			sound = torch.cat([left_zeros,sound,right_zeros],dim=0).to(self.device)
			loaded_sound_length = self.sound_length

		if loaded_sound_length > self.sound_length:

			start_index = torch.randint(0,(loaded_sound_length-self.sound_length)//2,(1,1))[0][0].item()
			end_index = start_index + self.sound_length
			sound = sound[start_index:end_index]

		sound = sound.unsqueeze(0)
		return sound

def save_sounds(path,sounds,sampling_rate):
	now_time = time.time()
	for i,sound in enumerate(sounds):
		sound = sound.squeeze(0)
		sound = sound.to('cpu').detach().numpy().copy()
		hash_string = hashlib.md5(str(now_time).encode()).hexdigest()
		file_path = os.path.join(path,f"generated_sound_{i}_{hash_string}.wav")
		print(file_path)
		sf.write(file_path,sound,sampling_rate,format="WAV")





