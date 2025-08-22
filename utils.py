from torchvision import transforms
import numpy as np
import torch
import logging
import os

class CustomizeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform=None):

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        self.A = data_dict["A"]
        self.X_n = data_dict["X_n"]
        self.H = data_dict["H"]
        self.X_e = data_dict["X_e"]
        self.I = data_dict["I"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.A)

    def __getitem__(self, index):
        A_ret = self.A[index]
        X_n_ret = self.X_n[index]
        H_ret = self.H[index]
        X_e_ret = self.X_e[index]
        I_ret = self.I[index]
        labels_ret = self.labels[index]

        if self.transform:
            A_ret = self.transform(A_ret)
            X_n_ret = self.transform(X_n_ret)
            H_ret = self.transform(H_ret)
            X_e_ret = self.transform(X_e_ret)
            I_ret = self.transform(I_ret)

        if not isinstance(A_ret, (dict)):
            A_ret = np.array(A_ret, dtype=np.float32)
            X_n_ret = np.array(X_n_ret, dtype=np.float32)
            H_ret = np.array(H_ret, dtype=np.float32)
            X_e_ret = np.array(X_e_ret, dtype=np.float32)
            I_ret = np.array(I_ret, dtype=np.float32)
            labels_ret = np.array(labels_ret, dtype=np.longlong)
        else:
            labels_ret = np.array(labels_ret, dtype=np.longlong)

        return A_ret, X_n_ret, H_ret, X_e_ret, I_ret, labels_ret

def create_save_dir(save_dir: str): 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    return save_dir

def get_logger(logger_name:str='DPS', save_dir:str='./', fname:str='run.log'):
    logger = logging.getLogger(name=logger_name)
    logger.setLevel(logging.INFO)
    
    if logger_name == 'BEST':
        formatter = logging.Formatter("\n%(asctime)s [%(name)s] >> %(message)s")
    else:
        formatter = logging.Formatter("%(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    if logger_name == '|':
        stream_handler.terminator = ' | '

    save_dir = create_save_dir(save_dir)
    file_handler = logging.FileHandler(filename= os.path.join(save_dir, fname))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    if logger_name == '|':
        file_handler.terminator = ' | '

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger