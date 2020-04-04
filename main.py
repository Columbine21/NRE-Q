import dataset
from config import opt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import PCNN
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def train(**kwargs):
    opt.parse(kwargs)
    if opt.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Loading training & Testing Dataset.
    DataModel = getattr(dataset, 'SEMData')
    train_data = DataModel('./dataset', train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    test_data = DataModel('./dataset', train=False)
    test_data_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    # criterion & optimizer (lr = opt.lr)
    model = PCNN(opt)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    best_acc = 0.0

    for epoch in range(opt.num_epochs):

        total_loss = 0.0

        for index, data in enumerate(train_data_loader):
            data = list(map(lambda x: x.to(device), data))

            model.zero_grad()
            out = model(data[:-1])
            loss = criterion(out, data[-1])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_avg_loss = total_loss / len(train_data_loader.dataset)
        acc, f1, eval_avg_loss, pred_y = eval(model, test_data_loader, opt.rel_num)
        if best_acc < acc:
            best_acc = acc
            write_result(opt.data+opt.model, pred_y)
            model.save(name=opt.data + "-" + opt.model)

        # toy_acc, toy_f1, toy_loss = eval(model, train_data_loader, opt.rel_num)
        print('Epoch {}/{}: train loss: {}; test accuracy: {}, test f1:{},  test loss {}'.format(
            epoch, opt.num_epochs, train_avg_loss, acc, f1, eval_avg_loss))
    print("*" * 30)
    print("the best acc: {};".format(best_acc))


def eval(model, test_data_loader, k):

    model.eval()
    if opt.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    avg_loss = 0.0
    pred_y = []
    labels = []

    for index, data in enumerate(test_data_loader):
        data = list(map(lambda x: x.to(device), data))
        out = model(data[:-1])
        loss = F.cross_entropy(out, data[-1])
        pred_y.extend(torch.max(out, 1)[1].data.cpu().numpy().tolist())
        labels.extend(data[-1].data.cpu().numpy().tolist())
        avg_loss += loss.data.item()

    size = len(test_data_loader.dataset)
    assert len(pred_y) == size and len(labels) == size, "predicted size not correct."
    f1_result = f1_score(labels, pred_y, average='micro')
    accuracy = accuracy_score(labels, pred_y)
    model.train()
    return accuracy, f1_result, avg_loss / size, pred_y


def write_result(model_name, pred_y):
    
    if model_name.startswith('SEM'):
        with open("./semeval/result.txt", "w") as out:
            start = 8001
            end = start + len(pred_y)
            for index in range(start, end):
                out.write("{}\t{}\n".format(index, pred_y[index - start]))
    else:
        raise NameError("evaluate dataset not support.")


if __name__ == "__main__":
    train()

