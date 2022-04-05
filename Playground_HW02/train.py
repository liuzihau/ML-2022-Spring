import numpy as np

import torch
import torch.nn as nn
# optimizer wrapper
from optim.opt_lookahead import Lookahead

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter

# For Progress Bar
from tqdm import tqdm


def trainer(train_loader, validation_loader, model, config, device, train_len, validation_len):
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    if config["lookahead"] == 1:
        optimizer = Lookahead(optimizer)

    best_acc = 0.0
    for epoch in range(config["num_epoch"]):
        train_acc = 0.0
        train_loss = 0.0

        # training
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            if config['model_name'] == 'bidirection_lstm_model':
                features = features.view(-1, config['concat_nframes'], config['input_dim'])

            if config["loss"] == 'sequence':
                outputs = model(features)
                output_flatten = outputs.view(-1, 41)
                labels_flatten = labels.view(-1)
                loss = criterion(output_flatten, labels_flatten)
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if config["loss"] == "sequence":
                outputs = outputs[:, config['concat_nframes'] // 2, :].view(outputs.shape[0], -1)
                labels = labels[:, config['concat_nframes'] // 2]
            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
            train_loss += loss.item()

        # validation
        if validation_len > 0:
            val_acc = 0.0
            val_loss = 0.0
            if config['lookahead'] == 1:
                optimizer._backup_and_load_cache()
            model.eval()  # set the model to evaluation mode
            with torch.no_grad():
                for i, batch in enumerate(tqdm(validation_loader)):
                    features, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    if config['model_name'] == 'bidirection_lstm_model':
                        features = features.view(-1, config['concat_nframes'], config['input_dim'])
                    if config["loss"] == "sequence":
                        outputs = model(features)
                        output_flatten = outputs.view(-1, 41)
                        labels_flatten = labels.view(-1)
                        loss = criterion(output_flatten, labels_flatten)
                    else:
                        outputs = model(features)
                        loss = criterion(outputs, labels)

                    if config["loss"] == "sequence":
                        outputs = outputs[:, config['concat_nframes'] // 2, :].view(outputs.shape[0], -1)
                        labels = labels[:, config['concat_nframes'] // 2]
                    _, val_pred = torch.max(outputs, 1)
                    val_acc += (
                            val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                    val_loss += loss.item()

                print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                    epoch + 1, config["num_epoch"], train_acc / train_len, train_loss / len(train_loader),
                    val_acc / validation_len, val_loss / len(validation_loader)
                ))

                # if the model improves, save a checkpoint at this epoch
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), config["model_path"])
                    print('saving model with acc {:.3f}'.format(best_acc / validation_len))
                if config['lookahead'] == 1:
                    optimizer._clear_and_load_backup()
        else:
            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, config["num_epoch"], train_acc / train_len, train_loss / len(train_loader)
            ))

    # if not validating, save the last epoch
    if validation_len == 0:
        torch.save(model.state_dict(), config["model_path"])
        print('saving model at last epoch')


def predict(test_loader, config, model, device):
    prediction = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch
            features = features.to(device)
            if config['model_name'] == 'bidirection_lstm_model':
                features = features.view(-1, config['concat_nframes'], config['input_dim'])
            if config["loss"] == "sequence":
                outputs = model(features)
                outputs = outputs[:, config['concat_nframes'] // 2, :].view(outputs.shape[0], -1)
            else:
                outputs = model(features)

            _, test_prediction = torch.max(outputs, 1)  # get the index of the class with the highest probability
            prediction = np.concatenate((prediction, test_prediction.cpu().numpy()), axis=0)
    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(prediction):
            f.write('{},{}\n'.format(i, y))
