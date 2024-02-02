# Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration

This repository contains the code for the following articles:
- [Spectral Norm of Convolutional Layers with Circular and Zero Paddings](https://arxiv.org/abs/2402.00240) by Blaise Delattre, [Quentin Barthélemy](https://github.com/qbarthelemy), Alexandre Allauzen
- [Efficient Bound of Lipschitz Constant for Convolutional Layers by Gram Iteration](https://arxiv.org/abs/2305.16173) published at ICML 2023 by Blaise Delattre, [Quentin Barthélemy](https://github.com/qbarthelemy), [Alexandre Araujo](https://github.com/araujoalexandre) and Alexandre Allauzen.

Gram iteration is a deterministic method to compute spectral norm in quadratic convergence.
It exhibits SOTA results on GPU regarding spectral norm computations.

## This repository

### Outline
* `bounds.py` contains code for different spectral norm bounds. 

* `note_book_test_gram_iteration.ipynb` contains some examples of spectral norm bound computations for different methods on dense and convolutional layers.

* `train_local.py` contains code to launch a training. Start a default configuration run  `python train_local.py --bound delattre2023 --bound_n_iter 6 --lr 0.1 --r 0.1`

## Installation

Experiences were done using pytorch-cuda=11.7

`git clone https://github.com/blaisedelattre/lip4conv.git`
