import os
import torch

import networkx as nx
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import CustomizeDataset
from collections import Counter
import csv

class ADNI_UNC(object):
    def __init__(self, config):
        self.config = config
        self.data_folder = os.path.join(config["DATASET"]["DATA_ROOT"], "new_adni" if config["DATASET"]["TYPE"] != 'old' else "adni")
        self.data_matrix_folder = os.path.join(self.data_folder, "AD-Data/AD-Data")
        self.unc_labels_file_path = os.path.join(self.data_folder, "DT_thickness.xlsx" if config["DATASET"]["TYPE"] != 'old' else "DataTS.csv")

        self.features_list = ['DT_thickness.xlsx', 'Amyloid_SUVR.xlsx', 'FDG_SUVR.xlsx']
        self.feature_name = self.config["DATASET"]["NODE_FEATURE"]
        self.labels_dict = {'AD':0, 'CN':1, 'EMCI':2, 'LMCI':3, 'SMC':5}
        self.subject_map = self.get_roi_feature("../data/new_adni", self.features_list, self.labels_dict)

        # load data
        self.unc_data, self.unc_labels, self.node_feature = self.get_unc_matrix_data_labels()
        self.edge_num = 0
        print("[data & label loaded]")

        # get common edges which are > threshold
        self.threshold = config["DATASET"]["THRESHOLD"]
        self.adj_index = self.edge_processing()

        self.inc_matrix = self.incidence_matrix()
        self.L_0_data = self.graph_laplacian_data()
        self.L_1_data = self.hodge_laplacian_data()
        self.edge_weight_data = self.get_edge_weights()

        self.unique_labels = None
        self.CN_index, self.SMC_index, self.EMCI_index, self.LMCI_index, self.AD_index = (self.merge5to4groups())

    def list_filenames(self):
        return [f for f in os.listdir(self.data_matrix_folder) if os.path.isfile(os.path.join(self.data_matrix_folder, f))]
    
    def edge_processing(self):
        A_mean = np.mean(self.unc_data, axis=0)
        adj_index = A_mean > self.threshold
        self.unc_data = self.unc_data * adj_index
        self.edge_num = adj_index.sum() // 2
        print("Number of Edges:", self.edge_num)

        return adj_index

    def get_unc_id_labels(self):
        unc_labels_path = self.unc_labels_file_path

        if self.config["DATASET"]["TYPE"] == 'old':
            reader = csv.reader(open(unc_labels_path, "r"), delimiter=",")
            x = list(reader)
            data_labels_matrix = np.array(x)

            subject_id_col = np.array(data_labels_matrix[1:, 0]).reshape((-1, 1))
            label_col = np.array(data_labels_matrix[1:, 4]).reshape((-1, 1))
        else:
            reader = pd.read_excel(unc_labels_path, engine='openpyxl')
            data_labels_matrix = reader.to_numpy()

            subject_id_col = np.array(data_labels_matrix[1:, 0]).reshape((-1, 1))
            label_col = np.array(data_labels_matrix[1:, 2]).reshape((-1, 1))

        id_labels = np.append(subject_id_col, label_col, axis=1)

        return id_labels
    
    ### Preprocess the feature file by sorting and removing unnecessary columns
    def preprocess_feature_table(self, data_path, filename):
        
        feature = pd.read_excel(data_path + '/' + filename, engine='openpyxl')
        
        ### Sorting
        if 'Subject' in feature.columns:
            subjects_num = feature['Subject'].copy()
            
            for i, subject in enumerate(subjects_num):
                subjects_num[i] = int(subject[1:])
            
            feature['subject_num'] = subjects_num
            feature = feature.sort_values(by=['PTID', 'subject_num'], kind='stable')
            feature.drop(['subject_num'], axis=1, inplace=True)
        else:
            feature = feature.sort_values(by=['PTID'], kind='stable')
        
        feature.reset_index(drop=True, inplace=True)
        
        inspection_num = feature.groupby('PTID').size().to_list()
        PTID_list = list(feature['PTID'])

        inspection_index = []
        for inspection in inspection_num:
            for i in range(inspection):
                inspection_index.append(i)

        inspection_index_str = list(map(str, inspection_index))
        
        ID_list_tmp = list(zip(PTID_list, inspection_index_str))
        ID_list = list(map(lambda x:x[0] + '_' + x[1], ID_list_tmp))
        
        if filename == 'DT_thickness.xlsx':
            feature.insert(1, 'ID', ID_list)
            feature.insert(3, 'VISCODE', inspection_index)
        elif filename == 'Amyloid_SUVR.xlsx' or filename == 'FDG_SUVR.xlsx' or filename == 'Tau_SUVR.xlsx':
            feature.drop(['SCAN', 'EXAMDATE', 'AGE', 'PTGENDER', 'PTEDUCAT', 'PTRACCAT', 'PTETHCAT', 'PTMARRY', 'APOE4'], axis=1, inplace=True) # Remove unnecessary columns
            feature.insert(0, 'ID', ID_list)
            feature.insert(2, 'VISCODE', inspection_index)
        
        return feature

    ### Get dictionary for subject with ROI features
    def get_roi_feature(self, data_path, features_list, labels_dict):
        all_features = [] # List for storing preprocessed tables
        all_features_PTID = [] # List for storing PITD of preprocessed tables
        
        for filename in features_list:
            feature = self.preprocess_feature_table(data_path, filename) # Preprocess feature file
            
            PTID_list = list(feature['PTID'])
            DX_list = list(feature['DX'])

            DX_dict = {}
            for ptid, dx in zip(PTID_list, DX_list):
                if ptid in DX_dict and DX_dict[ptid][0] != dx:
                    DX_dict[ptid] = [-1]
                else:
                    DX_dict[ptid] = [dx]
            
            for i, ptid in enumerate(PTID_list):
                if DX_dict[ptid][0] == -1:
                    feature.drop(labels=i, inplace=True)

            feature.sort_index(inplace=True)
            
            all_features.append(feature)
            all_features_PTID.append(list(feature['PTID']))
        
        PTID_common_set = set()
        for i, ptid_list in enumerate(all_features_PTID):
            if i == 0:
                PTID_common_set = set(ptid_list)
            else:
                PTID_common_set = PTID_common_set & set(ptid_list)
        
        PTID_common_list = list(PTID_common_set)

        all_features_num = len(all_features)
        
        common_inspection = [[] for _ in range(all_features_num)]
        common_inspection_max = [] 
        
        for ptid in PTID_common_list:
            max_count = 0
            
            for i, feature in enumerate(all_features):
                PTID_inspection_num = list(feature['PTID']).count(ptid)
                common_inspection[i].append(PTID_inspection_num)
                max_count = max(max_count, PTID_inspection_num)

            common_inspection_max.append(max_count)

        common_feature = []
        common_inspection_dict = []
        common_inspection_max_dict = {}
        
        for i, feature in enumerate(all_features):
            common_feature.append(feature[feature['PTID'].isin(PTID_common_list)])
            common_inspection_dict.append({ptid:inspection for ptid, inspection in zip(PTID_common_list, common_inspection[i])})

        common_inspection_max_dict = {ptid:inspection for ptid, inspection in zip(PTID_common_list, common_inspection_max)}

        for ptid in PTID_common_list:
            for i, inspection in enumerate(common_inspection_dict):
                if inspection[ptid] < common_inspection_max_dict[ptid]:
                    last_inspection_index = inspection[ptid] - 1
                    copy_object_id = ptid + '_' + str(last_inspection_index)
                    
                    for inspection_index in range(inspection[ptid], common_inspection_max_dict[ptid]):
                        new_id = ptid + '_' + str(inspection_index)
                        copy_object = common_feature[i].loc[common_feature[i].ID == copy_object_id].copy()
                        copy_object['ID'] = new_id
                        common_feature[i] = pd.concat([common_feature[i], copy_object], axis=0)
        
        for i in range(all_features_num):
            common_feature[i].set_index('ID', inplace=True)
            common_feature[i].sort_index(inplace=True)
        
        ### Leave only the necessary columns in the feature file lastly
        subject = None
        for feature in common_feature:
            if 'Subject' in feature.columns:
                subject = feature['Subject']
                feature.drop(['Subject'], axis=1, inplace=True)

        common_feature[0].insert(0, "Subject", subject)
        labels = common_feature[0]['DX'].map(labels_dict)
        common_feature[0].drop(['DX'], axis=1, inplace=True)
        
        ### Create a dictionary with ROI feature and label as value and subject as key
        subject_map = {}
        for i, subject in enumerate(common_feature[0]['Subject']):
            if not pd.isna(labels[i]):
                subject_map[subject] = [torch.stack([torch.tensor(x.iloc[i, 3:], dtype=torch.float) for x in common_feature]), labels.iloc[i]] # loading thickness along with other features
                # subject_map[subject] = [torch.tensor([common_feature[1].iloc[i, 3:]], dtype=torch.float), labels[i]] # loading only other features

        return subject_map

    def load_each_unc_matrix(self, filename):
        brain_matrix = np.loadtxt(filename)
        return (brain_matrix + brain_matrix.T) / 2

    def get_unc_matrix_data_labels(self):
        feature_name = self.feature_name
        if os.path.exists(f'../node_feature_{feature_name}.npy'):
            return (
                np.load(f'../adni_adj_{feature_name}.npy', allow_pickle=True),
                np.load(f'../adni_label_{feature_name}.npy', allow_pickle=True),
                np.load(f'../node_feature_{feature_name}.npy', allow_pickle=True)
            )

        matrix_dir = self.data_matrix_folder

        id_labels_alignment = self.get_unc_id_labels()
        class_list = np.unique(id_labels_alignment[:, 1])
        
        id_of_interest = id_labels_alignment[np.where(id_labels_alignment[:,1] == 'CN')]
        id_of_interest = np.append(id_of_interest, id_labels_alignment[np.where(id_labels_alignment[:,1] == 'AD')], axis=0)
        id_of_interest = np.append(id_of_interest, id_labels_alignment[np.where(id_labels_alignment[:,1] == 'EMCI')], axis=0)
        id_of_interest = np.append(id_of_interest, id_labels_alignment[np.where(id_labels_alignment[:,1] == 'LMCI')], axis=0)
        id_of_interest = np.append(id_of_interest, id_labels_alignment[np.where(id_labels_alignment[:,1] == 'SMC')], axis=0)
        
        print("total number of interest group: ",id_of_interest.shape)

        self.unique_labels = class_list
        brain_matrix_filenames = self.list_filenames()

        init_file = brain_matrix_filenames[0]

        init_matrix = np.expand_dims(self.load_each_unc_matrix(os.path.join(matrix_dir, init_file)), axis=0)

        node_features = np.zeros((1,160,3))
        unc_labels = np.zeros((1, 1))

        for file_index in range(len(brain_matrix_filenames)):

            file_name = brain_matrix_filenames[file_index]
            file_name_list = file_name.split("_")
            if self.config["DATASET"]["TYPE"] == 'old': subject_id = file_name_list[0]
            else: subject_id = file_name_list[-1]

            if (subject_id not in id_of_interest[:,0]) or (subject_id not in self.subject_map):
                continue

            brain_matrix_each = self.load_each_unc_matrix(os.path.join(matrix_dir, file_name))
            barin_matrix_each_ext = np.expand_dims(brain_matrix_each, axis=0)
            init_matrix = np.append(init_matrix, barin_matrix_each_ext, axis=0)

            for id_label_index in range(id_labels_alignment.shape[0]):
                if id_labels_alignment[id_label_index, 0] == subject_id:
                    unc_label_each = np.argwhere(class_list == id_labels_alignment[id_label_index, 1])
                    unc_labels = np.append(unc_labels, unc_label_each, axis=0)
                    break
            node_features_each = np.array(self.subject_map[subject_id][0]).T
            node_features_each_ext = np.expand_dims(node_features_each, axis=0)
            node_features = np.append(node_features, node_features_each_ext, axis=0)

            if file_index % 100 == 0:
                print("{}, {}, {}".format(file_index, file_name, unc_label_each[0, 0]))

        if self.config["DATASET"]["NUM_DATA"] == 0:
            unc_data_update = init_matrix[1:]
            unc_labels_update = unc_labels[1:].reshape(-1).astype(np.int_)
            node_features = node_features[1:]
        else:
            unc_data_update = init_matrix[1:self.config["DATASET"]["NUM_DATA"]]
            unc_labels_update = unc_labels[1:self.config["DATASET"]["NUM_DATA"]].reshape(-1).astype(np.int_)
            node_features = node_features[1:self.config["DATASET"]["NUM_DATA"]]

        print("unc_data_update: {}".format(unc_data_update.shape))
        print("unc_labels: {}".format(unc_labels_update.shape))
        print("node_features: {}".format(node_features.shape))

        np.save('../adni_adj_'+feature_name, unc_data_update)
        np.save('../adni_label_'+feature_name, unc_labels_update)
        np.save('../node_feature_'+feature_name, node_features)
        
        return unc_data_update, unc_labels_update, node_features

    def node_degree(self):
        nd = np.sum(self.unc_data, axis=1)
        nd = nd.reshape(self.unc_data.shape[0], self.unc_data.shape[1], 1)
        return nd
    
    def incidence_matrix(self):
        I = torch.zeros((self.unc_data.shape[0], self.config["DATASET"]["NUM_ROI"], int(self.edge_num)))

        adj_matrix = nx.from_numpy_matrix(self.adj_index)
        inc_matrix = nx.incidence_matrix(adj_matrix, oriented=True).todense()
        inc_matrix = np.squeeze(np.asarray(inc_matrix))
        inc_matrix = torch.from_numpy(inc_matrix)
        
        for i in range(self.unc_data.shape[0]):
            I[i,:,:] = inc_matrix

        return I.numpy()
    
    def graph_laplacian_data(self):
        L_0 = np.zeros((self.unc_data.shape[0], self.config["DATASET"]["NUM_ROI"], self.config["DATASET"]["NUM_ROI"]))

        graph_lap = self.inc_matrix[0] @ (self.inc_matrix[0].T)

        for i in range(self.unc_data.shape[0]):
            L_0[i,:,:] = graph_lap

        return L_0

    def hodge_laplacian_data(self):
        H = np.zeros((self.unc_data.shape[0], int(self.edge_num), int(self.edge_num)))

        hodge_lap = self.inc_matrix[0].T @ (self.inc_matrix[0])

        for i in range(self.unc_data.shape[0]):
            H[i,:,:] = hodge_lap

        return H
    
    def get_edge_weights(self):
        X_e = torch.zeros((self.unc_data.shape[0], int(self.edge_num), 1))
        self.adj_index = torch.tensor(self.adj_index)
        self.adj_index = torch.triu(self.adj_index, diagonal=1)

        for a in range(self.unc_data.shape[0]):
            temp = self.unc_data[a,:,:]
            temp = torch.tensor(temp)
            temp = torch.triu(temp, diagonal=1)
            
            edge_weight = temp[self.adj_index]
            
            if self.config["DATASET"]["NORMALIZE"]: 
                X_e[a,:,0] = edge_weight / np.max(edge_weight)
            else: 
                X_e[a,:,0] = edge_weight

        return X_e.numpy()

    def merge5to4groups(self):
        # class_list:  ['AD' 'CN' 'EMCI' 'LMCI' 'MCI' 'SMC']

        AD_index = np.where(self.unc_labels == 0)[0]
        LMCI_index = np.where(self.unc_labels == 3)[0]
        CN_index = np.where(self.unc_labels == 1)[0]
        EMCI_index = np.where(self.unc_labels == 2)[0]
        SMC_index = np.where(self.unc_labels == 5)[0]

        return (
            CN_index,
            SMC_index,
            EMCI_index,
            LMCI_index,
            AD_index,
        )

class UniDataset:
    def __init__(self, config):
        self.config = config
        print("==>> self.config: ", self.config)

        if self.config["DATASET"]["CV"]:
            self.num_folds = self.config["DATASET"]["NUM_FOLD"]

        self.train_transform = transforms.Compose([transforms.ToTensor()])
        self.test_transform = transforms.Compose([transforms.ToTensor()])
        self.edge_num = 0

    def get_datasets(self):

        adni_class = ADNI_UNC(self.config)
        self.edge_num = adni_class.edge_num

        A = adni_class.L_0_data
        X_n = adni_class.node_feature

        H = adni_class.L_1_data
        X_e = adni_class.edge_weight_data

        I = adni_class.inc_matrix

        data_dict = {}

        data_dict["A"] = A
        data_dict["I"] = I
        data_dict["H"] = H

        X_n_CN = X_n[adni_class.CN_index]
        X_n_SMC = X_n[adni_class.SMC_index]
        X_n_EMCI = X_n[adni_class.EMCI_index]
        X_n_LMCI = X_n[adni_class.LMCI_index]
        X_n_AD = X_n[adni_class.AD_index]

        data_dict["X_n_CN"] = X_n_CN
        data_dict["X_n_SMC"] = X_n_SMC
        data_dict["X_n_EMCI"] = X_n_EMCI
        data_dict["X_n_LMCI"] = X_n_LMCI
        data_dict["X_n_AD"] = X_n_AD

        X_e_CN = X_e[adni_class.CN_index]
        X_e_SMC = X_e[adni_class.SMC_index]
        X_e_EMCI = X_e[adni_class.EMCI_index]
        X_e_LMCI = X_e[adni_class.LMCI_index]
        X_e_AD = X_e[adni_class.AD_index]

        data_dict["X_e_CN"] = X_e_CN
        data_dict["X_e_SMC"] = X_e_SMC
        data_dict["X_e_EMCI"] = X_e_EMCI
        data_dict["X_e_LMCI"] = X_e_LMCI
        data_dict["X_e_AD"] = X_e_AD

        return data_dict


class UniDataloader:
    def __init__(self, config):
        super(UniDataloader, self).__init__()

        self.config = config

        self.dataset_class = UniDataset(config)

        self.all_data_dict = self.dataset_class.get_datasets()

    def get_dataloader(self, split, batch_size=None):
        assert split in ("train", "val", "test", "cv"), "Unknown split '{}'".format(
            split
        )

        if split == "cv":

            from sklearn.model_selection import KFold

            trian_dataloader_list = []
            val_dataloader_list = []

            kfold = KFold(n_splits=self.config["DATASET"]["NUM_FOLD"])

            CN_X_n_train_list = []
            CN_X_n_val_list = []
            CN_X_e_train_list = []
            CN_X_e_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["X_n_CN"])
            ):
                each_fold_X_n_CN_train = self.all_data_dict["X_n_CN"][train_idx]
                each_fold_X_n_CN_val = self.all_data_dict["X_n_CN"][valid_idx]
                CN_X_n_train_list.append(each_fold_X_n_CN_train)
                CN_X_n_val_list.append(each_fold_X_n_CN_val)
                each_fold_X_e_CN_train = self.all_data_dict["X_e_CN"][train_idx]
                each_fold_X_e_CN_val = self.all_data_dict["X_e_CN"][valid_idx]
                CN_X_e_train_list.append(each_fold_X_e_CN_train)
                CN_X_e_val_list.append(each_fold_X_e_CN_val)

            SMC_X_n_train_list = []
            SMC_X_n_val_list = []
            SMC_X_e_train_list = []
            SMC_X_e_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["X_n_SMC"])
            ):
                each_fold_X_n_SMC_train = self.all_data_dict["X_n_SMC"][train_idx]
                each_fold_X_n_SMC_val = self.all_data_dict["X_n_SMC"][valid_idx]
                SMC_X_n_train_list.append(each_fold_X_n_SMC_train)
                SMC_X_n_val_list.append(each_fold_X_n_SMC_val)
                each_fold_X_e_SMC_train = self.all_data_dict["X_e_SMC"][train_idx]
                each_fold_X_e_SMC_val = self.all_data_dict["X_e_SMC"][valid_idx]
                SMC_X_e_train_list.append(each_fold_X_e_SMC_train)
                SMC_X_e_val_list.append(each_fold_X_e_SMC_val)

            EMCI_X_n_train_list = []
            EMCI_X_n_val_list = []
            EMCI_X_e_train_list = []
            EMCI_X_e_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["X_n_EMCI"])
            ):
                each_fold_X_n_EMCI_train = self.all_data_dict["X_n_EMCI"][train_idx]
                each_fold_X_n_EMCI_val = self.all_data_dict["X_n_EMCI"][valid_idx]
                EMCI_X_n_train_list.append(each_fold_X_n_EMCI_train)
                EMCI_X_n_val_list.append(each_fold_X_n_EMCI_val)
                each_fold_X_e_EMCI_train = self.all_data_dict["X_e_EMCI"][train_idx]
                each_fold_X_e_EMCI_val = self.all_data_dict["X_e_EMCI"][valid_idx]
                EMCI_X_e_train_list.append(each_fold_X_e_EMCI_train)
                EMCI_X_e_val_list.append(each_fold_X_e_EMCI_val)

            LMCI_X_n_train_list = []
            LMCI_X_n_val_list = []
            LMCI_X_e_train_list = []
            LMCI_X_e_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["X_n_LMCI"])
            ):
                each_fold_X_n_LMCI_train = self.all_data_dict["X_n_LMCI"][train_idx]
                each_fold_X_n_LMCI_val = self.all_data_dict["X_n_LMCI"][valid_idx]
                LMCI_X_n_train_list.append(each_fold_X_n_LMCI_train)
                LMCI_X_n_val_list.append(each_fold_X_n_LMCI_val)
                each_fold_X_e_LMCI_train = self.all_data_dict["X_e_LMCI"][train_idx]
                each_fold_X_e_LMCI_val = self.all_data_dict["X_e_LMCI"][valid_idx]
                LMCI_X_e_train_list.append(each_fold_X_e_LMCI_train)
                LMCI_X_e_val_list.append(each_fold_X_e_LMCI_val)

            AD_X_n_train_list = []
            AD_X_n_val_list = []
            AD_X_e_train_list = []
            AD_X_e_val_list = []
            for fold, (train_idx, valid_idx) in enumerate(
                kfold.split(self.all_data_dict["X_n_AD"])
            ):
                each_fold_X_n_AD_train = self.all_data_dict["X_n_AD"][train_idx]
                each_fold_X_n_AD_val = self.all_data_dict["X_n_AD"][valid_idx]
                AD_X_n_train_list.append(each_fold_X_n_AD_train)
                AD_X_n_val_list.append(each_fold_X_n_AD_val)
                each_fold_X_e_AD_train = self.all_data_dict["X_e_AD"][train_idx]
                each_fold_X_e_AD_val = self.all_data_dict["X_e_AD"][valid_idx]
                AD_X_e_train_list.append(each_fold_X_e_AD_train)
                AD_X_e_val_list.append(each_fold_X_e_AD_val)

            for fold_i in range(self.config["DATASET"]["NUM_FOLD"]):
                each_train_zero_X_n = np.array(AD_X_n_train_list[fold_i])
                each_train_one_X_n = np.array(CN_X_n_train_list[fold_i])
                each_train_two_X_n = np.array(EMCI_X_n_train_list[fold_i])
                each_train_three_X_n = np.array(LMCI_X_n_train_list[fold_i])
                each_train_five_X_n = np.array(SMC_X_n_train_list[fold_i])

                each_train_zero_X_e = np.array(AD_X_e_train_list[fold_i])
                each_train_one_X_e = np.array(CN_X_e_train_list[fold_i])
                each_train_two_X_e = np.array(EMCI_X_e_train_list[fold_i])
                each_train_three_X_e = np.array(LMCI_X_e_train_list[fold_i])
                each_train_five_X_e = np.array(SMC_X_e_train_list[fold_i])

                each_train_zero_label = np.zeros(each_train_zero_X_n.shape[0])
                each_train_one_label = np.ones(each_train_one_X_n.shape[0])
                each_train_two_label = np.full(each_train_two_X_n.shape[0],2)
                each_train_three_label = np.full(each_train_three_X_n.shape[0],3)
                each_train_five_label = np.full(each_train_five_X_n.shape[0],4)

                each_train_data_X_n = np.append(each_train_zero_X_n,each_train_one_X_n,axis=0)
                each_train_data_X_n = np.append(each_train_data_X_n,each_train_two_X_n,axis=0)
                each_train_data_X_n = np.append(each_train_data_X_n,each_train_three_X_n,axis=0)
                each_train_data_X_n = np.append(each_train_data_X_n,each_train_five_X_n,axis=0)

                each_train_data_X_e = np.append(each_train_zero_X_e,each_train_one_X_e,axis=0)
                each_train_data_X_e = np.append(each_train_data_X_e,each_train_two_X_e,axis=0)
                each_train_data_X_e = np.append(each_train_data_X_e,each_train_three_X_e,axis=0)
                each_train_data_X_e = np.append(each_train_data_X_e,each_train_five_X_e,axis=0)

                each_train_data_A = self.all_data_dict["A"][:each_train_data_X_n.shape[0]]
                each_train_data_H = self.all_data_dict["H"][:each_train_data_X_n.shape[0]]
                each_train_data_I = self.all_data_dict["I"][:each_train_data_H.shape[0]]

                each_train_label = np.append(each_train_zero_label,each_train_one_label,axis=0)
                each_train_label = np.append(each_train_label,each_train_two_label,axis=0)
                each_train_label = np.append(each_train_label,each_train_three_label,axis=0)
                each_train_label = np.append(each_train_label,each_train_five_label,axis=0)

                print(Counter(each_train_label))
                each_train_dataset_dict = {}
                each_train_dataset_dict["A"] = each_train_data_A
                each_train_dataset_dict["X_n"] = each_train_data_X_n
                each_train_dataset_dict["H"] = each_train_data_H
                each_train_dataset_dict["X_e"] = each_train_data_X_e
                each_train_dataset_dict["I"] = each_train_data_I
                each_train_dataset_dict["labels"] = each_train_label

                # validation
                each_val_zero_X_n = np.array(AD_X_n_val_list[fold_i])
                each_val_one_X_n = np.array(CN_X_n_val_list[fold_i])
                each_val_two_X_n = np.array(EMCI_X_n_val_list[fold_i])
                each_val_three_X_n = np.array(LMCI_X_n_val_list[fold_i])
                each_val_five_X_n = np.array(SMC_X_n_val_list[fold_i])

                each_val_zero_X_e = np.array(AD_X_e_val_list[fold_i])
                each_val_one_X_e = np.array(CN_X_e_val_list[fold_i])
                each_val_two_X_e = np.array(EMCI_X_e_val_list[fold_i])
                each_val_three_X_e = np.array(LMCI_X_e_val_list[fold_i])
                each_val_five_X_e = np.array(SMC_X_e_val_list[fold_i])

                each_val_zero_label = np.zeros(each_val_zero_X_n.shape[0])
                each_val_one_label = np.ones(each_val_one_X_n.shape[0])
                each_val_two_label = np.full(each_val_two_X_n.shape[0],2)
                each_val_three_label = np.full(each_val_three_X_n.shape[0],3)
                each_val_five_label = np.full(each_val_five_X_n.shape[0],4)

                each_val_data_X_n = np.append(each_val_zero_X_n,each_val_one_X_n,axis=0)
                each_val_data_X_n = np.append(each_val_data_X_n,each_val_two_X_n,axis=0)
                each_val_data_X_n = np.append(each_val_data_X_n,each_val_three_X_n,axis=0)
                each_val_data_X_n = np.append(each_val_data_X_n,each_val_five_X_n,axis=0)

                each_val_data_X_e = np.append(each_val_zero_X_e,each_val_one_X_e,axis=0)
                each_val_data_X_e = np.append(each_val_data_X_e,each_val_two_X_e,axis=0)
                each_val_data_X_e = np.append(each_val_data_X_e,each_val_three_X_e,axis=0)
                each_val_data_X_e = np.append(each_val_data_X_e,each_val_five_X_e,axis=0)

                each_val_data_A = self.all_data_dict["A"][:each_val_data_X_n.shape[0]]
                each_val_data_H = self.all_data_dict["H"][:each_val_data_A.shape[0]]
                each_val_data_I = self.all_data_dict["I"][:each_val_data_H.shape[0]]

                each_val_label = np.append(each_val_zero_label,each_val_one_label,axis=0)
                each_val_label = np.append(each_val_label,each_val_two_label,axis=0)
                each_val_label = np.append(each_val_label,each_val_three_label,axis=0)
                each_val_label = np.append(each_val_label,each_val_five_label,axis=0)

                print(Counter(each_val_label))
                each_val_dataset_dict = {}
                each_val_dataset_dict["A"] = each_val_data_A
                each_val_dataset_dict["X_n"] = each_val_data_X_n
                each_val_dataset_dict["H"] = each_val_data_H
                each_val_dataset_dict["X_e"] = each_val_data_X_e
                each_val_dataset_dict["I"] = each_val_data_I
                each_val_dataset_dict["labels"] = each_val_label

                # create dataloader
                train_dataset = CustomizeDataset(
                    each_train_dataset_dict,
                    transform=self.dataset_class.train_transform
                )
                val_dataset = CustomizeDataset(
                    each_val_dataset_dict, 
                    transform=self.dataset_class.test_transform
                )

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=self.config["TRAIN"]["BATCH_SIZE"],
                    # batch_size=len(train_dataset),
                    shuffle=True,
                    drop_last=False,
                    num_workers=self.config["DATASET"]["WORKERS"],
                    pin_memory=self.config["DATASET"]["PIN_MEMORY"],
                )

                val_dataloader = DataLoader(
                    val_dataset,
                    # batch_size=self.config["TRAIN"]["BATCH_SIZE"],
                    batch_size=len(val_dataset),
                    shuffle=False,
                    drop_last=False,
                    num_workers=self.config["DATASET"]["WORKERS"],
                    pin_memory=self.config["DATASET"]["PIN_MEMORY"],
                )

                print("train_dataloader: {}".format(len(train_dataloader)))
                print("val_dataloader: {}".format(len(val_dataloader)))

                trian_dataloader_list.append(train_dataloader)
                val_dataloader_list.append(val_dataloader)

            return trian_dataloader_list, val_dataloader_list