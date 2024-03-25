<p align="center"><img src="./one_disentanglement_hyper_selection/results/xife_s5b50m40
/xife_s5b50m40_reconstruct_traverse.png" alt="visual_characteristics" style="display: block; margin-left: auto; margin-right: auto; width: 100%; text-align: center;" > </p>

## Market Structure Mapping with Visual Characteristics
Code to replicate results in, "Market Structure Mapping with Visual Characteristics."  

If you find this work useful, please use the following citation information in plaintext or BibTeX format:

Sisodia, A, Kumar, V and Burnap, A (2024). Market Structure Mapping with Visual Characteristics. Marketing Science (under review).

```
@article{sisodia2024marketstructure,
  title= Market Structure Mapping with Visual Characteristics},
  author={Sisodia, A, Kumar, V and Burnap, A},
  journal={Marketing Science (under review)},
  volume={XXX},
  number={XXX},
  pages={XXX},
  year={2024},
  publisher={XXX}
}
```
This code will reproduce Figures and Tables of the paper.

## Installation
#### Libraries
This codebase was built using the Pytorch neural net framework.  However, it also depends on over 100 other Python packages, as well as GPU libraries including CUDA runtimes and cuDNN kernel support. We can not guarantee this code will run and replicate unless the same environment and library versions are used.  To install these packages (including the same library versions), please use the following:

    conda env create -f marketstructure_env.yml

#### Data
To download the dataset, please use the following links. Copy these files to `./one_disentanglement_hyper_selection/data/cars/`, `./three_disentanglement_hyper_selection/data/cars/`, and `./unsup_disentanglement_hyper_selection/data/cars/`.

* #### [Watches Dataset](https://www.dropbox.com/scl/fo/akj3w8pat0lg1fa4ax480/h?rlkey=5d4ykq5br3kzkwarhi4ld4na8&dl=0)

## Replication Steps

#### Step 1: Run BLP Demand Model

1. Change directory to `./blp_before_disentanglement` and estimate the BLP demand model by executing `python pyblp_code.py`.
2. Execute `Rscript helper_script.R` to produce `exp_python_image_table.csv`.
3. Move  `exp_python_image_table.csv` to `../dataset_creation`.

#### Step 2: Dataset Creation

1. Change directory to `./dataset_creation`.
2. Unzip `jpeg_files.zip`.
3. Execute `python npz_file_creation.py`.

#### Step 3: Grid Search for Hyperparamaters

1. Go to `./one_disentanglement_hyper_selection` and run disentanglement model with a unique $\lambda_1$, $\lambda_2$, and supervisory signal with 10 different seeds. Vary $\lambda_1$, $\lambda_2$, and supervisory signal combination. For example, in the below command, the seed is set to 1, $\lambda_1$=50, $\lambda_2$=20, and the supervisory signal is price. The model name is `price_s5b50m20`. 

```
python main.py --sup_signal1 price -s 5 --name price_s5b50m20 --btcvae-B 50 --btcvae-M 20
```

In the above command, seed, $\lambda_1$, and $\lambda_2$ is a scalar value. This codebase, specific to one supervisory signal, supports the following set of discrete supporting signals. Using any other name will result in an error.

```
price
xife
```

2. Go to `./three_disentanglement_hyper_selection/` and run disentanglement model with a unique $\lambda_1$, and $\lambda_2$ with 10 different seeds. Vary $\lambda_1$, and $\lambda_2$. For example, in the below command, the seed is set to 1, $\lambda_1$=50, $\lambda_2$=20, and the supervisory signal is a comboination of hpwt, mpg and space. The model name is `hpwt_mpg_space_s5b50m20`.

```
python main.py -s 5 --name hpwt_mpg_space_s5b50m20 --btcvae-B 50 --btcvae-M 20
```

2. Go to `./unsup_disentanglement_hyper_selection/` and run disentanglement model with a unique $\lambda_1$ with 10 different seeds. Vary $\lambda_1$ alone. For example, in the below command, the seed is set to 1, $\lambda_1$=50, $\lambda_2$=0, and the supervisory signal is a comboination of hpwt, mpg and . The model name is `hpwt_mpg_space_s5b50m20`. 

```
python main.py -s 5 --name unsup_s5b50m0 --btcvae-B 50 --btcvae-M 0
```

In the above command, seed, $\lambda_1$ is a scalar value and $\lambda_2$ is fixed at 0. This codebase is specific to no supervisory signals.

All the python commands will create a directory `results/<model-name>/` which will contain:

* **model.pt**: The model at the end of training.
* **specs.json**: The parameters used to run the program (default and modified with CLI).
* **train_losses.csv**: All (sub-)losses computed during training on the train and validation dataset.
* **test_losses.log**: All (sub-)losses computed at the end of training on the test1 and test2 dataset. 
* **filename_test1.csv**: filenames of all watches in the test1 dataset. 
* **filename_test2.csv**: filenames of all watches in the test2 dataset. 
* **filename_train.csv**: filenames of all watches in the train dataset. 
* **mean_params_test1.csv**: mean visual characteristics of all cars in the test1 dataset. 
* **mean_params_test2.csv**: mean visual characteristics of all cars in the test2 dataset. 
* **mean_params_train.csv**: mean visual characteristics of all cars in the train dataset. 

#### Step 4: Model Selection & UDR Calculation

Select the value of $\lambda_1$ and $\lambda_2$ for each supervisory signal at which the average supervised loss across 10 seeds on the test1 dataset is lowest. 

1. Go to `./one_disentanglement_hyper_selection/results` and execute `./mv_script.sh`.
2. Go to `./three_disentanglement_hyper_selection/results` and execute `./mv_script.sh`.
3. Go to `./unsup_disentanglement_hyper_selection/results` and execute `./mv_script.sh`.
4. Go to `./one_model_selection` and execute `cp ../one_disentanglement_hyper_selection/results/*/*csv`.
5. Go to `./three_model_selection` and execute `cp ../three_disentanglement_hyper_selection/results/*/*csv`.
6. Go to `./unsup_model_selection` and execute `cp ../unsup_disentanglement_hyper_selection/results/*/*csv`.
7. Go to `./one_model_selection` and execute `Rscript val_loss.R price` and `Rscript val_loss.R xife` to find the optimal hyperparameters for a particular supervisory signal.
8. Go to `./three_model_selection` and execute `Rscript val_loss.R hpwt_mpg_space` to find the optimal hyperparameters for a particular supervisory signal.
9. Go to `./one_model_selection` and execute `Rscript r_script_all.R price` and `Rscript r_script_all.R xife`.
10. Go to `./three_model_selection` and execute `Rscript r_script_all.R hpwt_mpg_space`.
11. Go to `./unsup_model_selection` and execute `Rscript r_script_all.R unsup`, `Rscript r_script_all.R vae`, and ``Rscript r_script_all.R ae`.
12. Calculate UDR corresponding to make-model fixed effects by executing `Rscript udr_calculation.R xife` from the `one_model_selection` directory. [ **Table X in the paper** ]
13. Calculate UDR corresponding to price by executing `Rscript udr_calculation.R price` from the `one_model_selection` directory. [ **Table X in the paper** ]
14. Calculate UDR corresponding to hpwt, mpg and space by executing `Rscript udr_calculation.R hpwt_mpg_space` from the `three_model_selection` directory. [ **Table X in the paper** ]
15. Calculate UDR corresponding to the unsupervised $\beta$-TCVAE by executing `Rscript udr_calculation.R unsup` from the `unsup_model_selection` directory. [ **Table X in the paper** ]
16. Calculate UDR corresponding to plain-vanilla VAE by executing `Rscript udr_calculation.R vae` from the `unsup_model_selection` directory. [ **Table X in the paper** ]
17. Calculate UDR corresponding to plain-vanilla AE by executing `Rscript udr_calculation.R ae` from the `unsup_model_selection` directory. [ **Table X in the paper** ]

#### Step 4: Poaterior Traversal Generation

1. Execute the following commands from `one_disentanglement_hyper_selection`. 
```
python main_viz.py --name xife_s5b50m40 -s 5
```

This will produce `./one_disentanglement_hyper_selection/results/xife_s5b50m40/xife_s5b50m40_reconstruct_traverse.png'. [ **Figure Y in the paper** ]

2. Execute the following commands from `unsup_disentanglement_hyper_selection`.
   
```
python main_viz.py --name unsup_s5b50m0 -s 5
python main_viz.py --name vae_s5 -s 5
python main_viz.py --name ae_s5 -s 5
```

This will produce `./unsup_disentanglement_hyper_selection/results/unsup_s5b50m0/unsup_s5b50m0_reconstruct_traverse.png` [ **Figure Y in the paper** ]; `./unsup_disentanglement_hyper_selection/results/vae_s5/vae_s5_reconstruct_traverse.png` [ **Figure Y in the paper** ]; `./unsup_disentanglement_hyper_selection/results/ae_s5/ae_s5_reconstruct_traverse.png` [ **Figure Y in the paper** ]

## Computing Resources

We used 64G GPU to run each model instance of 200 epoch, 64 batch size, 5e-4 learning rate. Each model run takes 4 minutes. We train 10 random seeds * 3 supervisory signal combinations * 25*15 grid values = 11,250 models. This would mean 750 hours of model training.

## Citation

Feel free to use this code for your research. If you find this code useful, please use the following citation information in plaintext or BibTeX format:

Sisodia, A, Burnap, A, and Kumar, V (2024). Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis. Journal of Marketing Research (under review).

```
@article{sisodia2024disentangelment,
  title= Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis},
  author={Sisodia, A, Burnap, A, and Kumar, V},
  journal={Journal of Marketing Research (under review)},
  volume={XXX},
  number={XXX},
  pages={XXX},
  year={2024},
  publisher={XXX}
}
```

## Acknowledgments
Portions of this codebase were built on elements from the following open source projects, please see their respective licenses and cite accordingly:
* [disentangling-vae](https://github.com/YannDubs/disentangling-vae)
