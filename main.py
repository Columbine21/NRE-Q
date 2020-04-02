import dataset
from config import opt
from torch.utils.data import DataLoader


if __name__ == "__main__":
    DataModel = getattr(dataset, "SEMData")
    train_data = DataModel('./dataset', train=True)
    train_data_loader = DataLoader(train_data, 32, shuffle=True, num_workers=4)

    test_data = DataModel('./dataset', train=False)
    test_data_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    print('train data: {}; test data: {}'.format(len(train_data), len(test_data)))

    for i, input in enumerate(train_data_loader):
        if i > 0:
            break
        word_features, left_pf, right_pf, lexical_features, labels = input
        print(word_features.shape, left_pf.shape, right_pf.shape, lexical_features.shape, labels.shape)
