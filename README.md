# CNN4Jets - Convolutional neural network (CNN) for jet classification
## Quickstart:
To create and train a model, execute the **main.py** file in the shell:

`$ python3 main.py`

Now the following things will happen:

1. The model Loads the data from the specified txt files, e.g.:

`'input_paths'           : {'ttbar'    : './conv_out_ttbar.txt',
                           'qcdJets'  : './conv_out_qcdJets.txt'}`.
                           
2. The model will create a CNN network.
3. The model creates a dataset from the loaded txt files.
4. A test set of images will be saved as numpy arrays in **./model/image_data** for each of the classes.
5. Furthermore these images will be plottet and saved to **./model/images**.
6. Then a training loop will start.
7. The model will be saved in a **model** file in home folder **./**




`


## Todo:

-[x] install Madgraph, ROOT & Delphes

-[x] Make calorimater images from delphes output

-[x] Write params card for constit2images

-[x] Preprocessing of the jet images

-[x] Train/test split in Dataset

-[x] Write model architecture

-[x] Write training loop

-[x] Write validation

-[] Additional learning rate scheduler
