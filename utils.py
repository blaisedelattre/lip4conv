
def get_n_iter_epoch(epoch):
    if epoch < 125:
        return 3
    elif 125 <= epoch < 150:
        return 4
    elif 150 <= epoch < 175:
        return 5
    elif 175 <= epoch:
        return 6
