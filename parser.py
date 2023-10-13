import argparse


def parse():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--bound",
        default="delattre2023_backward",
        type=str,
        help="type of bound",
        choices=[
            "delattre2023_backward",
            "ours_backward",
            "araujo2021",
            "sedghi2019",
            "singla2021",
            "ryu2019",
        ],
    )
    parser.add_argument(
        "--bound_n_iter",
        default=6,
        type=int,
        help="number of iteration for bound method",
    )
    parser.add_argument("--bs", default=256, type=int, help="batch size for training")
    parser.add_argument("--r", default=0.1, type=float, help="Lip conv penalty")
    parser.add_argument(
        "--threshold_reg",
        default=1.0,
        type=float,
        help="Threshold for regularization value",
    )
    parser.add_argument("--wd", default=0.0, type=float, help="weight decay")
    parser.add_argument("--epochs", default=200, type=int, help="number of epoch")
    parser.add_argument("--adaptative_bound_n_iter", action="store_true")
    args = parser.parse_args()
    return args


def get_name(args):
    res = "run"
    for arg in vars(args):
        res += "_" + str(arg) + str(getattr(args, arg))
    return res


if __name__ == "__main__":
    args = parse()
    print(get_name(args))
