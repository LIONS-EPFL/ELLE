# Efficient local linearity regularization to overcome catastrophic overfitting

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)[![Licence](https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge)](./LICENSE)[![OpenReview](https://img.shields.io/badge/OpenReview-8C1B13?style=for-the-badge)](https://openreview.net/forum?id=SZzQz8ikwg)



Official implementation in PyTorch of the **ICLR'24 paper "Efficient local linearity regularization to overcome catastrophic overfitting"**.

"One torch snippet is worth a thousand words"
    - A wise Deep Learner 

```python
# Get random triplet
bs = x.shape[0]
x_ab = x.repeat([2,1,1,1]) 
x_ab = x_ab + eps*(2*torch.rand(x_ab.shape, device = x.device) - 1)
alpha = torch.rand([bs,1,1,1],device = x.device)
x_c = (1-alpha)*x_ab[:bs] + alpha*x_ab[bs:]
alpha = alpha.squeeze()

# Forward pass
criterion = torch.nn.CrossEntropyLoss(reduction='none')
losses = criterion(model(torch.cat((x_ab,x_c),dim=0)), y.repeat([3]))

# Regularization term
mse = torch.nn.MSELoss()
lin_err = mse(losses[2*bs:], (1-alpha)*losses[:bs] + alpha*losses[bs:2*bs])
```
This snippet can be used to obtain our regularization term. `model` is a neural network, `x` are the batched inputs, `y` the batched labels and `eps` the perturbation budget.

### Instalation:
The dependencies of this project can be easily installed via:
```
conda env create -f environment.yml
```
If Autoattack evaluations are needed, it can be installed via:
```
pip install git+https://github.com/fra31/auto-attack
```

## Sample scripts
Sample scripts to train with LLR, CURE, N-FGSM, ELLE(-A) and GradAlign in the CIFAR10 dataset are available in `scripts`. We include our ImageNet code in `ImageNet`. The output of our training script is organized as:
```
- results_method
    - dataset_method_seedx_perteps_nEpoch
        - latest_checkpoint.pt
        - stats.csv
        - ...
```
where `stats.csv` stores the training, validation and test accuracies (clean and PGD-20) for every epoch. If one desires to compute the PGD-50-10 or AutoAttack accuracies, refer to `eval_pgd50-10_folder.py` and `eval_aa_folder.py` respectively.


## Cite as:

```
@inproceedings{Abad2024ELLE,
  author = {Abad Rocamora, Elias and Liu, Fanghui and Chrysos, Grigorios G and M. Olmos, Pablo and Cevher, Volkan},

  title = {Overcomming delayed catastrophic overfitting efficiently},

  booktitle = {International Conference on Learning Representations (ICLR)},

  year = {2024}
}
```
