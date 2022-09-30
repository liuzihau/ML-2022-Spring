import os
import time

from tqdm import tqdm
import json
import csv
from pathlib import Path
import torch
from torch import nn
from torch.optim import AdamW, Adam

import utils


def model_fn(config, batch, model, criterion, device):
    """Forward a batch through the model."""

    mels, labels = batch
    mels = mels.to(device)

    labels = labels.to(device)

    if config["criterion"] == "Additive_Margin_Softmax":
        loss, outs = model(mels, labels)
    else:
        outs = model(mels)
        loss = criterion(outs, labels)

    # Get the speaker id with the highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())

    return loss, accuracy


def valid(config, dataloader, model, criterion, device):
    """Validate on validation set."""

    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(config, batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(
            loss=f"{running_loss / (i + 1):.2f}",
            accuracy=f"{running_accuracy / (i + 1):.2f}",
        )

    pbar.close()
    model.train()

    return running_accuracy / len(dataloader)


def train_the_model(config, train_loader, valid_loader, speaker_num, train_no):
    model_path = config["model_path"]
    save_path = config['save_path']
    valid_steps = config['valid_steps']
    warmup_steps = config['warmup_steps']
    save_steps = config['save_steps']
    total_steps = config['total_steps']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    """Main function."""
    if config['model'] == "conformer":
        if config["criterion"] == "Additive_Margin_Softmax":
            from model.model3 import Classifier
            print('[Info]: Use model3')
        else:
            from model.model2 import Classifier
            print('[Info]: Use model2')
    else:
        from model.model import Classifier
        print('[Info]: Use model1')

    model = Classifier(config=config, n_spks=speaker_num).to(device)
    # model.load_state_dict(torch.load(model_path))

    if config["criterion"] == "Additive_Margin_Softmax":
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = utils.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(config, batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(config, valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, f"{save_path}_{train_no}.ckpt")
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


def predict_and_save_result(config, dataloader, i):
    data_dir = config['data_dir']
    model_path = config["model_path"]
    output_path = config["output_path"]

    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    speaker_num = len(mapping["id2speaker"])
    if config['model'] == "conformer":
        if config["criterion"] == "Additive_Margin_Softmax":
            from model.model3 import Classifier
            print('[Info]: Use model3')
        else:
            from model.model2 import Classifier
            print('[Info]: Use model2')
    else:
        from model.model import Classifier
    model = Classifier(config=config, n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(f"{model_path}_{i}.ckpt"))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            labels = torch.randint(0, 600, (mels.shape[0],))
            if config["criterion"] == "Additive_Margin_Softmax":
                _, outs = model(mels, labels)
            else:
                outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(f'{output_path}_{i}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)


def train_adaptation_model(config, src_data_loader, tgt_data_loader, speaker_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################
    if config['model'] == "conformer":
        if config["criterion"] == "Additive_Margin_Softmax":
            from model.model3 import Classifier, Discriminator
            print('[Info]: Use model3')
        else:
            from model.model2 import Classifier, Discriminator
            print('[Info]: Use model2')
    else:
        from model.model import Classifier
        print('[Info]: Use model1')

    # set train state for Dropout and BN layers
    src_encoder = Classifier(config=config, n_spks=speaker_num).to(device)
    src_encoder.load_state_dict(torch.load(f"{config['model_path']}_4.ckpt"))

    tgt_encoder = Classifier(config=config, n_spks=speaker_num).to(device)
    tgt_encoder.load_state_dict(torch.load(f"{config['model_path']}_4.ckpt"))

    discriminator = Discriminator(config=config).to(device)
    # setup criterion and optimizer
    criterion = nn.BCELoss().to(device)
    optimizer_tgt = Adam(tgt_encoder.parameters(), lr=1e-4)
    optimizer_critic = Adam(discriminator.parameters(), lr=1e-4)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################
    for run_number in range(500):
        print(f"[Info] Running version: [{run_number} / 100]")
        best_loss = 10000
        ###########################
        # 2.1 train discriminator #
        ###########################
        for epoch in range(config['adaptation_epoch']*3):
            # zip source and target data pair
            data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
            for step, ((src, _), (tgt, _)) in data_zip:
                src = src.to(device)
                tgt = tgt.to(device)

                src_encoder.eval().to(device)
                tgt_encoder.eval().to(device)
                discriminator.train().to(device)

                # zero gradients for optimizer
                optimizer_critic.zero_grad()

                # extract and concat features
                feat_src = src_encoder(src, feature_extractor=True).to(device)
                feat_tgt = tgt_encoder(tgt, feature_extractor=True).to(device)
                feat_concat = torch.cat((feat_src, feat_tgt), 0).to(device)

                # predict on discriminator
                pred_concat = discriminator(feat_concat.detach()).to(device)
                # prepare real and fake label
                label_src = torch.ones(feat_src.size(0)).float()
                label_tgt = torch.zeros(feat_tgt.size(0)).float()
                label_concat = torch.cat((label_src, label_tgt), 0).to(device)
                label_concat = torch.unsqueeze(label_concat, 1)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat)
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()
                pred_cls = [1 if b >= 0.5 else 0 for b in pred_concat.to('cpu').detach().numpy().reshape(-1)]
                pred_cls = torch.Tensor(pred_cls).to(device)
                label_concat = torch.squeeze(label_concat, 1)
                acc = (pred_cls == label_concat).float().mean()

                if float(loss_critic.item()) < best_loss:
                    print(f'[Info] save dis model with best loss : {loss_critic.item()}')
                    torch.save(discriminator.state_dict(),
                               os.path.join('./', f"ADDA-discriminator_{run_number}_best.pt"))
                    best_loss = float(loss_critic.item())

                #######################
                # print step info #
                #######################

                if (step + 1) % 250 == 0:
                    print("Epoch [{}/{}] Step [{}/{}]:"
                          "d_loss={:.5f} acc={:.5f}"
                          .format(epoch + 1,
                                  config['adaptation_epoch'],
                                  step + 1,
                                  len_data_loader,
                                  loss_critic.item(),
                                  acc.data))

        ############################
        # 2.2 train target encoder #
        ############################
        best_loss = 100000
        for epoch in range(config['adaptation_epoch']):
            # zip source and target data pair
            data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
            for step, ((src, _), (tgt, _)) in data_zip:
                src = src.to(device)
                tgt = tgt.to(device)
                src_encoder.eval()
                tgt_encoder.train()
                discriminator.eval()
                # zero gradients for optimizer
                optimizer_critic.zero_grad()
                optimizer_tgt.zero_grad()

                # extract and target features
                feat_tgt = tgt_encoder(tgt, feature_extractor=True).to(device)

                # predict on discriminator
                pred_tgt = discriminator(feat_tgt)

                # prepare fake labels
                label_tgt = torch.ones(feat_tgt.size(0)).float().to(device)
                label_tgt = torch.unsqueeze(label_tgt, 1)
                # compute loss for target encoder
                loss_tgt = criterion(pred_tgt, label_tgt)
                loss_tgt.backward()

                # optimize target encoder
                optimizer_tgt.step()
                pred_cls = [1 if b >= 0.5 else 0 for b in pred_tgt.to('cpu').detach().numpy().reshape(-1)]
                pred_cls = torch.Tensor(pred_cls).to(device)
                acc = (pred_cls == label_tgt).float().mean()

                if float(loss_tgt.item()) < best_loss:
                    print(f'[Info]save encoder model with best loss : {loss_tgt.item()}')
                    torch.save(discriminator.state_dict(),
                               os.path.join('./', f"ADDA-tgt_encoder_{run_number}_best.pt"))

                    best_loss = float(loss_tgt.item())
                #######################
                # 2.3 print step info #
                #######################

                if (step + 1) % 250 == 0:
                    print("Epoch [{}/{}] Step [{}/{}]:"
                          "g_loss={:.5f} acc={:.5f}"
                          .format(epoch + 1,
                                  config['adaptation_epoch'],
                                  step + 1,
                                  len_data_loader,
                                  loss_tgt.item(),
                                  acc.data))

    return tgt_encoder


def predict_and_save_result_with_adda(config, dataloader,tgt_encoder):
    data_dir = config['data_dir']
    model_path = config["model_path"]
    output_path = config["output_path"]

    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    mapping_path = Path(data_dir) / "mapping.json"
    mapping = json.load(mapping_path.open())
    speaker_num = len(mapping["id2speaker"])
    if config['model'] == "conformer":
        if config["criterion"] == "Additive_Margin_Softmax":
            from model.model3 import Classifier
            print('[Info]: Use model3')
        else:
            from model.model2 import Classifier
            print('[Info]: Use model2')
    else:
        from model.model import Classifier
    tgt_encoder.eval()
    print(f"[Info]: Finish creating tgt_encoder!", flush=True)
    model = Classifier(config=config, n_spks=speaker_num).to(device)
    model.load_state_dict(torch.load(f"{model_path}_{4}.ckpt"))
    model.eval()
    print(f"[Info]: Finish creating model!", flush=True)

    results = [["Id", "Category"]]
    for feat_paths, mels in tqdm(dataloader):
        with torch.no_grad():
            mels = mels.to(device)
            labels = torch.randint(0, 600, (mels.shape[0],))
            if config["criterion"] == "Additive_Margin_Softmax":
                stats = tgt_encoder(mels, labels, feature_extractor=True)
                _, outs = model(stats, labels, classify_only=True)
            else:
                outs = model(mels)
            preds = outs.argmax(1).cpu().numpy()
            for feat_path, pred in zip(feat_paths, preds):
                results.append([feat_path, mapping["id2speaker"][str(pred)]])

    with open(f'{output_path}_adda.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(results)
