## AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss

This repository is a fork of [AUTOVC: Zero-Shot Voice Style Transfer with Only Autoencoder Loss](https://github.com/auspicious3000/autovc). 

This repository provides a PyTorch implementation of AUTOVC, appropriately modified in order to make style transfer well also on languages other than English.

### Dependencies
- Python 3
- jupyter
- Numpy
- PyTorch >= v0.4.1
- wavenet_vocoder ```pip install wavenet_vocoder```
  for more information, please refer to https://github.com/r9y9/wavenet_vocoder
- librosa
- soundfile
- scipy
- tqdm
- matplotlib
- wavio
- spleeter for more information, please refer to https://github.com/deezer/spleeter
- ffmpeg

### Pre-trained models

| AUTOVC | Speaker Encoder | WaveNet Vocoder |
|----------------|----------------|----------------|
| [link](https://drive.google.com/file/d/1SZPPnWAgpGrh0gQ7bXQJXXjOntbh4hmz/view?usp=sharing)| [link](https://drive.google.com/file/d/1ORAeb4DlS_65WDkQN6LHx5dPyCM5PAVV/view?usp=sharing) | [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing) |


### 0.Convert Mel-Spectrograms

Download pre-trained AUTOVC model, and run the ```conversion.ipynb``` in the same directory.

The fast and high-quality hifi-gan v1 (https://github.com/jik876/hifi-gan) pre-trained model is now available [here.](https://drive.google.com/file/d/1n76jHs8k1sDQ3Eh5ajXwdxuY_EZw4N9N/view?usp=sharing)


### 1.Mel-Spectrograms to waveform

Download pre-trained WaveNet Vocoder model, and run the ```vocoder.ipynb``` in the same the directory.

Please note the training metadata and testing metadata have different formats.


### 2.Train model

We have included a small set of training audio files in the wav folder. However, the data is very small and is for code verification purpose only. Please prepare your own dataset for training.

1.Generate spectrogram data from the wav files: ```python make_spect.py```

2.Generate training metadata, including the GE2E speaker embedding (please use one-hot embeddings if you are not doing zero-shot conversion): ```python make_metadata.py```

3.Run the main training script: ```python main.py```

Converges when the reconstruction loss is around 0.0001.

## Train with new vocoder
```
python3.8 make_spect.py # create folder spmel
python3.8 make_spect_other_vocoder.py # create the folder spmel_other
CUDA_VISIBLE_DEVICES="0" python3.8 make_metadata.py --root-dir="./spmel" # create the spmel/train.pkl # use speaker encoder on /spmel
cp spmel/train.pkl spmel_other # copy the spmel/train.pkl into spmel_other/train.pkl
CUDA_VISIBLE_DEVICES="0" python3.8 main.py --data_dir="spmel_other" \
    --outfile-path="/home/super/Models/autovc_simple/generator.pth" \
    --num_iters 10000 --batch_size=6 --dim_neck 32 --dim_emb 256 --dim_pre 512 --freq 32
CUDA_VISIBLE_DEVICES="0" python3.8 test_audio.py



