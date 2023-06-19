# READ ME

Code for different spectral norm bounds is in `lip_conv/bounds.py`

Find in jupyter notebook `note_book_test_gram_iteration.ipynb` some examples of spectral norm bound computations for different methods on dense and convolutional layers.

To launch a training with default configuration run : 

`python train_local.py --bound ours_backward --bound_n_iter 6 --lr 0.1 --r 0.1`