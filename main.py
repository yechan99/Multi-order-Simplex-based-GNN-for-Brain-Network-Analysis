from dataloader import UniDataloader
from model import Ours
from utils import get_logger

import yaml
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support

import time
import datetime
import random
import torch.backends.cudnn as cudnn

def fix_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

def load_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_logger(config):
    best_logger = get_logger(logger_name='BEST', save_dir=config["SETTING"]["LOG_DIR"], fname=config["SETTING"]["LOG_FILE"])
    logger = get_logger(logger_name='|', save_dir=config["SETTING"]["LOG_DIR"], fname=config["SETTING"]["LOG_FILE"])
    logger.info(config)
    return best_logger, logger

def train_epoch(model, dataloader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    for A_train, X_train_n, H_train, X_train_e, I_train, y_train in dataloader:
        A_train, X_train_n, H_train, X_train_e, I_train, y_train = map(lambda x: x.to(device), [A_train, X_train_n, H_train, X_train_e, I_train, y_train])
        pred = model(X_train_n, X_train_e, A_train, H_train, I_train)
        loss = loss_function(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * A_train.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    correct = 0
    wrong = 0
    results = [0] * 5
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for A_val, X_val_n, H_val, X_val_e, I_val, y_val in dataloader:
            A_val, X_val_n, H_val, X_val_e, I_val, y_val = map(lambda x: x.to(device), [A_val, X_val_n, H_val, X_val_e, I_val, y_val])
            prediction = model(X_val_n, X_val_e, A_val, H_val, I_val)
            loss = loss_function(prediction, y_val)
            total_loss += loss.item() * A_val.size(0)
            preds = torch.argmax(prediction, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())
            for yhat, ypred in zip(y_val, preds):
                if yhat == ypred:
                    results[int(yhat)] += 1
                else:
                    wrong += 1

    total_loss /= len(dataloader.dataset)
    right = sum(results)
    accuracy = 100 * right / (right + wrong)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    precision, recall, f1 = precision * 100, recall * 100, f1 * 100

    return total_loss, accuracy, precision, recall, f1

def main():
    config = load_config("config.yaml")
    best_logger, logger = initialize_logger(config)
    fix_random_seed(config["SETTING"]["RANDOM_SEED"])

    dataloader_class = UniDataloader(config)
    train_dataloader_list, val_dataloader_list = dataloader_class.get_dataloader("cv")

    device = get_device()
    node_num = config["DATASET"]["NUM_ROI"]
    edge_num = int(dataloader_class.dataset_class.edge_num)

    total_results = [[], [], [], []] # acc / precision / recall / f1
    start = time.time()

    for i in range(config["DATASET"]["NUM_FOLD"]):
        logger.info(f"\n##### Fold {i + 1} #####")
        best_val_f1 = 0

        model = Ours(
            node_num,
            edge_num,
            config["DATASET"]["NODE_FEATURE_NUM"],
            1,
            config["TRAIN"]["HIDDEN_DIM"],
            config["DATASET"]["NUM_CLASSES"],
            0.5,
            config["TRAIN"]["NUM_GCN_LAYER"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=config["NET"]["LR"], weight_decay=config["NET"]["WEIGHT_DECAY"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config["NET"]["LR_FACTOR"], patience=config["NET"]["LR_PAT"], min_lr=0.000001
        )
        loss_function = nn.NLLLoss()

        train_dataloader = train_dataloader_list[i]
        val_dataloader = val_dataloader_list[i]

        for j in range(config["TRAIN"]["MAX_EPOCHS"]):
            logger.info(f"\nEpoch {j + 1}")
            logger.info(f"LR={optimizer.param_groups[0]['lr']}")

            train_loss = train_epoch(model, train_dataloader, loss_function, optimizer, device)
            scheduler.step(train_loss)

            logger.info(f"Train Loss: {train_loss:.5f}")

            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_dataloader, loss_function, device)
            logger.info(f"Val Loss: {val_loss:.5f}")
            logger.info(f"accuracy: {val_acc:.2f}")
            logger.info(f"precision: {val_prec:.2f}")
            logger.info(f"recall: {val_rec:.2f}")
            logger.info(f"f1: {val_f1:.2f}")

            elapsed_time = str(datetime.timedelta(seconds=time.time() - start)).split(".")[0]
            logger.info(f"{elapsed_time} sec")

            if best_val_f1 < val_f1:
                best_val_acc = val_acc
                best_val_prec = val_prec
                best_val_rec = val_rec
                best_val_f1 = val_f1
                best_logger.info(f"val Loss: {val_loss:.5f} | accuracy: {val_acc:.2f} | precision: {val_prec:.2f} | recall: {val_rec:.2f} | f1: {val_f1:.2f} |")

            if j + 1 == config["TRAIN"]["MAX_EPOCHS"]:
                total_results[0].append(best_val_acc) 
                total_results[1].append(best_val_prec)
                total_results[2].append(best_val_rec)
                total_results[3].append(best_val_f1)

    logger.info("\n#########   total_results   #########\n")
    total_results = np.array(total_results, dtype=object)
    logger.info(total_results)

    logger.info("Mean")
    for i in range(4):
        logger.info(f"{np.mean(total_results[i]):.2f}")

    logger.info("Std")
    for i in range(4):
        logger.info(f"{np.std(total_results[i]):.2f}")

if __name__ == "__main__":
    main()
