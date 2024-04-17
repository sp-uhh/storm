import torch
import torchaudio
from sgmse.model import DiscriminativeModel
from sgmse.backbones import NCSNpp
from sgmse.util.other  import pad_spec
# # Load the checkpoint
# checkpoint_path = '/export/home/lemercier/code/_public_repos/storm/.logs/denoiser_ncsnpp_vctk-reverb_epoch=113.ckpt'
# model = DiscriminativeModel.load_from_checkpoint(checkpoint_path)

# # Get the state_dict of the backbone
# backbone_state_dict = model.dnn.state_dict()

# # Save the state_dict as a .pt file
# output_path = '/export/home/lemercier/code/audiodps/ncsnppm_epoch=113.pt'
# torch.save(backbone_state_dict, output_path)


output_path = '/export/home/lemercier/code/audiodps/ncsnppm_epoch=113.pt'
istft_kwargs = {"n_fft": 510, "hop_length": 128, "win_length": 510, "center": True, "window": torch.hann_window(510).cuda()}
stft_kwargs = {**istft_kwargs, "return_complex": True}

if __name__ == "__main__":

    dnn = NCSNpp(discriminative=True)
    dnn.load_state_dict(torch.load(output_path))
    dnn.cuda()

    x, _ = torchaudio.load("/data/lemercier/databases/VCTK-Reverb/test/reverberant/p226/p226_156.wav")
    x = x[...,1*16000: 3*16000]

    X = torch.stft(x.cuda(), **stft_kwargs).unsqueeze(1)
    X = torch.sqrt(X.abs()) * torch.exp(1j * X.angle())
    X = pad_spec(X)
    Y = dnn(X)
    Y = torch.square(Y.abs()) * torch.exp(1j * Y.angle())
    y = torch.istft(Y.squeeze(1), **istft_kwargs).cpu()

    torchaudio.save("p226_156_denoised.wav", y, 16000)