from itertools import product
import os
import uuid

from parser import parse, get_name
from trainer import Trainer

os.environ["DATADIR"] = "./data"


def main():
    args = parse()
    name_args = get_name(args)
    id_exp = uuid.uuid4().hex
    evalfile = name_args + "_" + str(id_exp)
    args.evalfile = evalfile
    print("args", args)
    trainer = Trainer(args)
    trainer()


if __name__ == "__main__":
    main()
