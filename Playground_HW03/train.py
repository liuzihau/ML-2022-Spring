import numpy as np
import pandas as pd
import torch
from torch import nn
# This is for the progress bar.
from tqdm.auto import tqdm
from timm.data.mixup import Mixup


def train_the_model(model, loader, config, criterion, optimizer, split=0):
    # Initialize trackers, these are not parameters and should not be changed
    stale = 0
    best_acc = 0
    train_loader = loader["train"][split]
    mixup_args = {
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.,
        'prob': 1,
        'switch_prob': 0.5,
        'mode': 'batch',
        'label_smoothing': 0.1,
        'num_classes': 11
    }
    mixup_fn = Mixup(**mixup_args)
    for epoch in range(config["n_epochs"]):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()
        model = model.to(config["device"])

        # These are used to record information in training.
        train_loss = []
        train_accs = []
        i = 0
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            images, labels = batch
            images, labels = mixup_fn(images.to(config["device"]), labels.argmax(dim=-1).to(config["device"]))
            # images = images.half()
            # print(images.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(images.to(config["device"]))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(config["device"]))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(config["device"]).argmax(dim=-1)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)
            if (i + 1) % config['validation_iter'] == 0:
                train_loss = sum(train_loss) / config['validation_iter']
                train_acc = sum(train_accs) / config['validation_iter']

                # Print the information.
                print(
                    f"[ Train | {epoch + 1:03d}/{config['n_epochs']:03d} ] [ Iter | {i + 1:03d}/{len(train_loader)} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

                valid_accuracy_average = validate_the_model(model, loader['valid'][split], config, criterion, optimizer,
                                                            epoch,
                                                            best_acc)
                print(f"[ Valid | [ Iter | {i + 1:03d}/{len(train_loader)} ]")

                train_loss = []
                train_accs = []

                # save models
                if valid_accuracy_average > best_acc:
                    print(f"Best model found at epoch {epoch + 1}, saving model")
                    torch.save(model.state_dict(),
                               f"{config['_exp_name']}_best{split}.ckpt")  # only save best to prevent output memory exceed error
                    best_acc = valid_accuracy_average
                    stale = 0
                else:
                    stale += 1
                    if stale > config['patience']:
                        print(f"No improvement {config['patience']} consecutive epochs, early stopping")
                        break
            i += 1
            # Print the information.

        valid_accuracy_average = validate_the_model(model, loader['valid'][split], config, criterion, optimizer, epoch,
                                                    best_acc)

        # save models
        if valid_accuracy_average > best_acc:
            print(f"Best model found at epoch {epoch + 1}, saving model")
            torch.save(model.state_dict(),
                       f"{config['_exp_name']}_best.ckpt")  # only save best to prevent output memory exceed error
            best_acc = valid_accuracy_average
            stale = 0
        else:
            stale += 1
            if stale > config['patience']:
                print(f"No improvement {config['patience']} consecutive epochs, early stopping")
                break

    return best_acc


def validate_the_model(model, valid_loader, config, criterion, optimizer, epoch, best_acc):
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    if config['lookahead'] == 1:
        optimizer._backup_and_load_cache()
    valid_loss = []
    valid_accuracy = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):
        if config['ensemble'] == 1:
            imgs_set, labels = batch
        else:
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            if config['ensemble'] == 1:
                imgs_set = imgs_set.to(config["device"])
                logits = model(torch.squeeze(imgs_set[:, 0, :, :, :]))
            else:
                logits = model(imgs.to(config["device"]))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(config["device"]))

        # Compute the accuracy for current batch.
        if config['ensemble'] == 1:
            ensemble = imgs_set.shape[1]
            softmax = nn.Softmax()
            for idx in range(ensemble):
                logits = model(torch.squeeze(imgs_set[:, idx, :, :, :]).to(config["device"]))
                if idx == 0:
                    ensembled_logits = softmax(logits) * config['ensemble_ratio']
                else:
                    ensembled_logits += softmax(logits) * ((1 - config['ensemble_ratio']) / (ensemble - 1))
            acc = (ensembled_logits.argmax(dim=-1) == labels.to(config["device"]).argmax(dim=-1)).float().mean()
        else:
            acc = (logits.argmax(dim=-1) == labels.to(config["device"]).argmax(dim=-1)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accuracy.append(acc)
        # break
        if config['lookahead'] == 1:
            optimizer._clear_and_load_backup()

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss_average = sum(valid_loss) / len(valid_loss)
    valid_accuracy_average = sum(valid_accuracy) / len(valid_accuracy)

    # Print the information.
    print(
        f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss_average:.5f}, acc = {valid_accuracy_average:.5f}")

    # update logs
    if valid_accuracy_average > best_acc:
        with open(f"./{config['_exp_name']}_log.txt", "a"):
            print(
                f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss_average:.5f}, acc = {valid_accuracy_average:.5f} -> best")
    else:
        with open(f"./{config['_exp_name']}_log.txt", "a"):
            print(
                f"[ Valid | {epoch + 1:03d}/{config['n_epochs']:03d} ] loss = {valid_loss_average:.5f}, acc = {valid_accuracy_average:.5f}")

    return valid_accuracy_average


def do_prediction(model, config, test_loader, test_set):
    model.eval()
    prediction = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            if config['ensemble'] == 1:
                ensemble = data.shape[1]
                for idx in range(ensemble):
                    logits = model(torch.squeeze(data[:, idx, :, :, :]).to(config["device"]))
                    if idx == 0:
                        ensembled_logits = logits * config['ensemble_ratio']
                    else:
                        ensembled_logits += logits * ((1 - config['ensemble_ratio']) / (ensemble - 1))
                test_label = np.argmax(ensembled_logits.cpu().data.numpy(), axis=1)
            else:
                test_prediction = model(data.to(config["device"]))
                test_label = np.argmax(test_prediction.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

    # create test csv
    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    df = pd.DataFrame()
    df["Id"] = [pad4(i) for i in range(1, len(test_set) + 1)]
    df["Category"] = prediction
    df.to_csv("submission.csv", index=False)
