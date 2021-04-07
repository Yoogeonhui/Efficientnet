import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import RMSprop, AdamW
from torch.optim.lr_scheduler import ExponentialLR
import torchvision
from torch import nn
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

from model import EfficientNetConfig, EfficientNet
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import wandb

best_val_score = 0.0


def conf_to_name(conf:EfficientNetConfig):
    return f'{conf.depth} {conf.width} {conf.resolution}'


def CIFAR10_label_list():
    cur_dict = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    print([k for k in cur_dict])
    return [k for k in cur_dict]


def train(cur_model, run_name):
    optimizer = AdamW(cur_model.parameters(), lr=1e-4, weight_decay=0.99)
    scheduler = ExponentialLR(optimizer, 0.97)
    val = int(224 * cur_model.config.resolution + 5)//10 * 10
    print(val)

    transform = transforms.Compose(
        [transforms.Resize(val),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
    first = int(len(dataset) * 0.9)
    second = len(dataset) - first
    torch.manual_seed(0)
    train_dataset, val_dataset = random_split(dataset, [first, second])
    train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    cur_model.train()

    def run_validation():
        global best_val_score
        cur_model.eval()

        with torch.no_grad():
            acc_loss, acc_label, acc_pred = 0, [], []
            for X, y in val_dataloader:
                X, y = X.cuda(), y.cuda()
                pred = cur_model(X)
                loss = criterion(pred, y)
                acc_loss += loss.item() * X.shape[0]
                acc_label.extend(y.detach().cpu().tolist())
                acc_pred.append(pred.detach().cpu())
        res = np.concatenate(acc_pred, axis=0)
        armax = np.argmax(res, axis=1)
        acc_score = accuracy_score(acc_label, armax.tolist())
        f1score = f1_score(acc_label, armax.tolist(), average='macro')
        wandb.log({'val_loss': acc_loss/len(acc_label),
                   "val_conf_mat": wandb.plot.confusion_matrix(probs=res, y_true=np.array(acc_label),
                                                               class_names=CIFAR10_label_list()),
                   'val_acc': acc_score,
                   'val_f1': f1score})
        if best_val_score < acc_score:
            best_val_score = acc_score
            print(f'saved {run_name}, {acc_score}')
            torch.save(cur_model.state_dict(), run_name+'.pth')
        cur_model.train()

    for epoch in range(10):
        acc_loss, acc_label, acc_pred = 0, [], []

        for i, (X, y) in tqdm(enumerate(train_dataloader)):
            X, y = X.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = cur_model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            acc_loss += loss.item() * X.shape[0]
            acc_label.extend(y.detach().cpu().tolist())
            pred = torch.argmax(pred, dim=1)
            acc_pred.extend(pred.detach().cpu().tolist())
            if i % 10 == 0:
                wandb.log({'train_acc': accuracy_score(acc_label, acc_pred),
                           'train_f1': f1_score(acc_label, acc_pred, average='macro'),
                           'train_loss': acc_loss/len(acc_label)})
                acc_loss, acc_label, acc_pred = 0, [], []
        scheduler.step(int(epoch/2.4))

        run_validation()


def get_grid_list():
    ret = []
    interval = 0.1
    alpha = 1
    while alpha <= 2.0:
        beta = 1.0
        while alpha * (beta ** 2) <=2.0:
            gamma = (2 / (alpha * beta**2))**(1/2)
            if gamma >= 1.0:
                # gamma는 0.05 FLOATING
                alpha = int((alpha + 0.025) * 20)/20
                beta = int((beta + 0.025) * 20)/20
                gamma = int((gamma + 0.025) * 20)/20
                ret.append([alpha, beta, gamma])
            beta += interval
        alpha += interval
    print(len(ret))
    return ret


def main():
    global best_val_score
    gl = get_grid_list()
    for g in gl:
        best_val_score = 0.0
        conf = EfficientNetConfig(depth=g[0], width=g[1], resolution=g[2], num_classes=10)
        run_name = conf_to_name(conf)

        run = wandb.init(project='EfficientNet_small', reinit=True)
        run.name = run_name
        run.save()

        cur_model = EfficientNet(conf)
        cur_model.cuda()

        wandb.watch(cur_model)

        train(cur_model, run_name)
        print(run_name, best_val_score)
        run.finish()



if __name__ == '__main__':
    main()
