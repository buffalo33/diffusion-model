# Diffusion Model for images generation based on SDE
Here is an implementation of a Diffusion Model for images generation which is built on MNIST dataset. This Diffusion model was built using python and it's inspired from this articles:
- https://arxiv.org/pdf/2006.11239.pdf

- https://arxiv.org/pdf/2011.13456.pdf

and these githubs: 
- https://github.com/yang-song/score_sde

- https://github.com/hojonathanho/diffusion


## Dependencies:
 ````
 numpy, pandas, pytorch, Tensorboard
 ```` 
 
## What to execute ??

### Prerequisites
The Notebooks are meant to be executed in Google Colab linked to your Drive to import all the files. You can run it without Colab but in this case the commands of copy from your drive need to be ignore.
If you use Notebooks with Colab and Drive, please create a folder named `modified version` in your `Colab Notebooks` folder. The `modified version` must contain all the files of this repository.

### For Training:
- train.py
- or train.ipynb
### For Generating
- generate.py
- or generate.ipynb

To follow up the losses/ BDPs and the FIDs over epochs, we have used the tensorborad.

## Contributors :

  - ALJOHANI Renad
  
  - DJECTA Hibat_Errahmen
  
  - PRÉAUT Clément
  
