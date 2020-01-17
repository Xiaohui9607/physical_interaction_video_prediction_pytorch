from options import Options
from model import Model

def train():
    opt = Options().parse()
    model = Model(opt)

    model.train()


if __name__ == '__main__':
    train()