import os
import json
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


def load_data(datasets, ensemble_dataset, config, tfm):
    final_train_set = []
    for dataset in datasets:
        train_set = dataset(os.path.join(config["_dataset_dir"], "training"), config, tfm=tfm["train"])
        final_train_set.append(train_set)
    concat_train_set = ConcatDataset(final_train_set)
    train_loader = DataLoader(concat_train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                              pin_memory=True)
    if config['ensemble'] == 1:
        valid_set = ensemble_dataset(os.path.join(config["_dataset_dir"], "validation"), config, tfm=tfm["train"])
        valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                  pin_memory=True)
        test_set = ensemble_dataset(os.path.join(config["_dataset_dir"], "test"), config, tfm=tfm["train"])
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0,
                                 pin_memory=True)
    else:
        valid_set = datasets[0](os.path.join(config["_dataset_dir"], "validation"), config, tfm=tfm["test"])
        valid_loader = DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                  pin_memory=True)
        test_set = datasets[0](os.path.join(config["_dataset_dir"], "test"), config, tfm=tfm["test"])
        test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0,
                                 pin_memory=True)
    loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }
    return train_set, valid_set, test_set, loader


def load_data_from_folder(datasets, ensemble_dataset, config, tfm):
    loader = {
        "train": [],
        "valid": [],
        "test": []
    }
    for i in range(config["cross_valid"]):

        final_train_set = []
        train_paths = [x for x in os.listdir(config["_dataset_dir"]) if ("argument" in x and "test" not in x)]
        train_paths.append("all_origin_training_data")
        for dataset in datasets:
            for train_path in train_paths:
                train_set = dataset(os.path.join(config["_dataset_dir"], train_path), config, tfm=tfm["test"], split=i,mode='train')
                final_train_set.append(train_set)
        concat_train_set = ConcatDataset(final_train_set)
        loader['train'].append(
            DataLoader(concat_train_set, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
        )
        if config['ensemble'] == 1:
            # todo:ensemble logic not finished
            valid_set = ensemble_dataset(os.path.join(config["_dataset_dir"], "all_origin_training_data"), config, tfm=tfm["test"], split=i,mode='valid')
            loader['valid'].append(
                DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
            )
        else:
            valid_set = datasets[0](os.path.join(config["_dataset_dir"], "all_origin_training_data"), config, tfm=tfm["test"], split=i,mode='valid')
            loader['valid'].append(
                DataLoader(valid_set, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
            )
    if config['ensemble'] == 1:
        test_paths = [os.path.join(config["_dataset_dir"],x) for x in os.listdir(config["_dataset_dir"]) if ("argument" in x and "test" in x)]
        test_set = ensemble_dataset(test_paths, config, tfm=tfm["test"],mode='test')
        loader['test'].append(
            DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        )
    else:
        test_set = datasets[0](os.path.join(config["_dataset_dir"], "test"), config, tfm=tfm["test"],mode='test')
        loader['test'].append(
            DataLoader(test_set, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
        )

    return test_set, loader


if __name__ == "__main__":
    with open('./config/config.json', 'r', encoding="utf-8") as config_json:
        config = json.loads(config_json.read())
    debug_train_path = [x for x in os.listdir(config["_dataset_dir"]) if ("argument" in x and "test" not in x)]
    debug_train_path.append("training")
    print(debug_train_path)
