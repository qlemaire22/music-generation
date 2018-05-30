# Music-Generation

The implementation of the LSTM network was based on the work of Sigurður Skúli ([link](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5)) and the VAE implementation was based on the tutorial by François Chollet on Keras ([link](https://blog.keras.io/building-autoencoders-in-keras.html)).

## Requirements

- Python 3.6
- Keras 2.1.6
- Tensorflow (used as Keras's backend) 1.8.0
- Music21 5.1.0

The two last requirements are only necessary if you want to be able to listen to MIDI files directly with Python.

- midi2audio 0.1.1
- FluidSynth 1.1.1 (and at least one audio font)

## Dataset

In our experiments, we have used the Essen Folksong Collection ([link](http://kern.ccarh.org/browse?l=essen)). It is composed of more than 8 000 folksongs from all around the world. However, the collection consists mainly of songs from China and Germany (more than 7 000 songs). Therefore, we have a dataset that is very sparse and the generated songs without any constrain will be greatly influence by those two styles.

Each song is composed of one instrument playing a short melody (the average number of notes by file is 50). The notes are going from A2 to C7 and it makes a total of 76 different notes (flat and sharp included, C# is considered different than D-).

## How to reproduce?

Some of the information are still hardcoded in some files for now so some manipulations will be needed.

- First, download the dataset on your computer and change the absolut path toward the dataset in `config.py`.

- `python preprocessing.py` to parse the dataset and create new files containing the information we need.

- `python train.py` to train the LSTM network over the whole dataset.

- Then, you can generate memories of the network according to some constrains. The constrains are hardcoded in the file `generate_memories.py`, it will select each song that match one of the conditions in their file name to create the memories. Once you have written your condition, you also need to write the name of file containing the saved weights of the network and then run `python generate_memories.py`.

- Once the memories have been generated, you can train the VAE over the memories. First, you have to put the path toward the `.npy` containing the memories in `vae/vae_data.py`. After that, you can train the VAE with `python vae/vae_train.py`.

- Finally, rewrite you condition and the name of the weight file in `predict_specific.py` and then run it with `python predict_specific.py` to generate a song that matches the conditions.
