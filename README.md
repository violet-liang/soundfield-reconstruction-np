# Sound field reconstruction using neural processes with dynamic kernels

In our [paper](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-024-00333-x),we propose a new approach that parameterizes GPs using a deep neural network based on Neural Processes (NPs) to reconstruct the sound field. This method has the advantage of dynamically learning
kernels from simulated data using an attention mechanism, allowing for greater flexibility and adaptability to the acoustic properties of the sound field.

## Data Preparation
For training, We used 10,000 rooms and placed a variable number of microphones on a 32-by-32 grid within the frequency range of [30,500] Hz, which was evenly divided into 40 frequency bins. The dataset has the dimensions of [10000, 32, 32, 40].
For inference, the dataset has the dimensions of [1000, 32, 32, 40].

## Training
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py
```

## Inference:
Run using GPU:
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

## Citation
```
@article{liang2024sound,
  title={Sound field reconstruction using neural processes with dynamic kernels},
  author={Liang, Zining and Zhang, Wen and Abhayapala, Thushara D},
  journal={EURASIP Journal on Audio, Speech, and Music Processing},
  volume={2024},
  number={1},
  pages={13},
  year={2024},
  publisher={Springer}
}
```
