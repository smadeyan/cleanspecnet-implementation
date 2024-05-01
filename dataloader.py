import os
import json
import numpy as np
import random
from scipy.io.wavfile import read as wavread
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torchaudio

class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs.
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    """

    def __init__(self, root='./', subset='training', crop_length_sec=0):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["training", "testing"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset

        N_clean = len(os.listdir(os.path.join(root, 'clean')))
        N_noisy = len(os.listdir(os.path.join(root, 'noisy')))
        assert N_clean == N_noisy

        if subset == "training":
            self.files = [(os.path.join(root, 'clean', 'fileid_{}.wav'.format(i)),
                           os.path.join(root, 'noisy', 'fileid_{}.wav'.format(i))) for i in range(N_clean)]

        elif subset == "testing":
            sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
            # Assumig you're inside the CleanUNet Folder
            _p = os.path.join("../DNS-Challenge", 'datasets/test_set/synthetic/no_reverb')  # path for DNS

            clean_files = os.listdir(os.path.join(_p, 'clean'))
            noisy_files = os.listdir(os.path.join(_p, 'noisy'))

            clean_files.sort(key=sortkey)
            noisy_files.sort(key=sortkey)

            self.files = []
            for _c, _n in zip(clean_files, noisy_files):
                assert sortkey(_c) == sortkey(_n)
                self.files.append((os.path.join(_p, 'clean', _c),
                                   os.path.join(_p, 'noisy', _n)))
            self.crop_length_sec = 0

        else:
            raise NotImplementedError

    def __getitem__(self, n):
        fileid = self.files[n]
        clean_audio, sample_rate = torchaudio.load(fileid[0])
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        clean_audio, noisy_audio = clean_audio.squeeze(0), noisy_audio.squeeze(0)
        assert len(clean_audio) == len(noisy_audio)

        crop_length = int(self.crop_length_sec * sample_rate)
        assert crop_length < len(clean_audio)

        # random crop
        if self.subset != 'testing' and crop_length > 0:
            start = np.random.randint(low=0, high=len(clean_audio) - crop_length + 1)
            clean_audio = clean_audio[start:(start + crop_length)]
            noisy_audio = noisy_audio[start:(start + crop_length)]

        transform = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=256)

        clean_stft = transform(clean_audio)
        noisy_stft = transform(noisy_audio)

        clean_stft, noisy_stft = clean_stft.unsqueeze(1), noisy_stft.unsqueeze(1)
        return (clean_stft, noisy_stft, fileid)

    def __len__(self):
        return len(self.files)
    

def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec)
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)

    return dataloader


if __name__ == '__main__':
    with open("/config/DNS-large-full.json") as f:
        data = f.read()

    cleanspecnet_config = json.loads(data)
    trainset_config = cleanspecnet_config["trainset_config"]

    trainloader = load_CleanNoisyPairDataset(**trainset_config, subset='training', batch_size=config['batch_size'], num_gpus=1)
    testloader = load_CleanNoisyPairDataset(**trainset_config, subset='testing', batch_size=config['batch_size'], num_gpus=1)

    print(len(trainloader), len(testloader))

    for clean_audio, noisy_audio, fileid in trainloader:
        clean_audio = clean_audio.cuda()
        noisy_audio = noisy_audio.cuda()
        print(clean_audio.shape, noisy_audio.shape, fileid)
        break