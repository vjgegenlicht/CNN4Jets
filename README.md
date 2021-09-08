# CNN4Jets - Convolutional neural network (CNN) for jet classification
## Quickstart:
To create and train a model, execute the *main.py* file in the shell:

`$ python3 main.py`

Now the following things will happen:

1. The model loads the data from the specified txt files, e.g.:

`'input_paths'           : {'ttbar'    : './conv_out_ttbar.txt',
                           'qcdJets'  : './conv_out_qcdJets.txt'}`.
                           
2. The model will create a CNN network.
3. The model creates a dataset from the loaded txt files.
4. A test set of images will be saved as numpy arrays in: **./model/image_data** for each of the classes.
5. Furthermore these images will be plottet and saved to: **./model/images**.
6. Then a training loop will start.
7. The model will be saved in a *model* file in the folder: **./model**

Now a model exists, that can be used for the classification itself. Creating and training a model will not be done on the computer in the museum. Instead, a model file and all additional data will be provided beforehand. 

To classify a image, two parser arguments exist:

`--model=<path>` and `--classify=<path>`.

Using these arguments, one can then choose an image file from **./model/image_data** and classify it. For example, we can classify an image of the *ttbar* dataset:

`$ python3 main.py --model=./model/model --classify=./model/image_data/ttbar/X_test_i.npy`.

The index i must be replaced by an existing image in the **./model/image_data/ttbar** folder. The model then predicts the probabilities of the image to be in each of the classes and saves these probabilities as a numpy array in a *tmp_prob.npy* file in the **./model** folder




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
