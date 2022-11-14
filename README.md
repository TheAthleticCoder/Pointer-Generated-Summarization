# **Pointer_Gen_Summary**
In this repository, we implement the paper: **Get To The Point: Summarization with Pointer-Generator Networks.** 

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)
[![forthebadge](https://forthebadge.com/images/badges/fixed-bugs.svg)](https://forthebadge.com)

-----
## **Objectives**

In this repository, we aim to accomplish the following tasks:
1. The input to the model will be a text paragraph from which you need to create summaries.
2. Build a Pointer Generator Network architecture from scratch using PyTorch (You may use any other
framework of your choice, but do not use an off-the-shelf library implementation!)
3. The encoder should be a BiLSTM, whereas the decoder should be a single LSTM layer
4. Create attention distribution with the help of the decoder state and encoder hidden states
5. The context vector is a weighted sum of encoder hidden states as per the respective attention distribution
6. For each decoder timestep calculate generation probability pgen
7. Use a weighted sum of vocabulary distribution and attention distribution to obtain a final distribution
to make the final prediction
8. use ROUGE Metric to evaluate the model

The Pointer network can be considered a simple extension of the attention model. It is a hybrid of an
Attention Model and a pointer network. Words are generated from a fixed vocabulary and are copied
by pointing.

-----

## **File Structure**

1. `data_reduction.py` contains the code to choose a suitable subset of the entire dataset we have used.
2. `summary_gen.py` We store the decoder outputs and summaries generated in a csv file. This file opens that csv and prints out the summaries in a viewable fashion as well as prints ROUGE scores.
3. `extra` folder containing training logs, ipynb format for the codes and other extra files.
4. `pgn_summzarization.py` Single commented out and explained code file which handles datasets, constructs data loaders, builds the model from as given and starts training.
5. `README` itz what you reading right now ^_^ 
-----

## **Execution**

The code is executed by:
```py
python3 <filename>.py
```
When the model is run, the code snippet:

```py
torch.save(model.state_dict(), 'model.pt')
```
stores the best parameters of the model which gives the lowest validation loss. The code by itself calls the model back for testing purposes using:

```py
model.load_state_dict(torch.load('model.pt'))
```
If the model has been successfully loaded, it returns `<All keys matched successfully>`.



-----


