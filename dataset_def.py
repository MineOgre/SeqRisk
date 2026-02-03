from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import numpy as np


def get_Normalization(X, mask, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    ## convert X to float
    X = X.astype(np.float64)
    X[mask == 0] = np.nan

    means = np.zeros(num_Feature)
    stds = np.zeros(num_Feature)
    mins = np.zeros(num_Feature)
    maxs = np.zeros(num_Feature)

    if norm_mode == 'standard':  # zero mean unit variance
        for j in range(num_Feature):
            mean = np.nanmean(X[:, j])
            std = np.nanstd(X[:, j])
            means[j] = mean
            stds[j] = std

            if std != 0:
                X[:, j] = (X[:, j] - mean) / std
            else:
                X[:, j] = X[:, j] - mean

        X[np.isnan(X)] = 0
        return X, means, stds

    elif norm_mode == 'normal':  # min-max normalization
        for j in range(num_Feature):
            min_val = np.nanmin(X[:, j])
            max_val = np.nanmax(X[:, j])
            mins[j] = min_val
            maxs[j] = max_val

            X[:, j] = (X[:, j] - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else X[:, j]

        X[np.isnan(X)] = 0
        return X, mins, maxs

    else:
        print("INPUT MODE ERROR!")
        return None, None, None

class PhysionetDataset(Dataset):
    """
    Dataset definition for the Physionet Challenge 2012 dataset.
    """

    def __init__(self, data_file, root_dir, transform=None):
        data = np.load(os.path.join(root_dir, data_file))
        self.data_source = data['data_readings'].reshape(-1, data['data_readings'].shape[-1])
        self.label_source = data['outcome_attrib'].reshape(-1, data['outcome_attrib'].shape[-1])
        self.mask_source = data['data_mask'].reshape(-1, data['data_mask'].shape[-1])
        self.label_mask_source = data['outcome_mask'].reshape(-1, data['outcome_mask'].shape[-1])
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        patient_data = self.data_source[idx, :]
        patient_data = torch.Tensor(np.array(patient_data))

        mask = self.mask_source[idx, :]
        mask = np.array(mask, dtype='uint8')

        label = self.label_source[idx, :]
        label[8] = label[8] - 24
        label_mask = self.label_mask_source[idx, :]

        label = torch.Tensor(np.concatenate((label, label_mask)))

        if self.transform:
            patient_data = self.transform(patient_data)

        sample = {'data': patient_data, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDataset(Dataset):
    """
    Dataset definition for the rotated MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 784.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array([digit])

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class RotatedMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the rotated MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 28 x 28.
    """

    def __init__(self, data_file, label_file, root_dir, mask_file=None, transform=None):

        data = np.load(os.path.join(root_dir, data_file))
        label = np.load(os.path.join(root_dir, label_file))
        self.data_source = data.reshape(-1, data.shape[-1])
        self.label_source = label.reshape(label.shape[0], -1).T
        if mask_file is not None:
            self.mask_source = np.load(os.path.join(root_dir, mask_file))
        else:
            self.mask_source = np.ones_like(self.data_source)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source[idx, :]
        digit = np.array(digit)
        digit = digit.reshape(28, 28)
        digit = digit[..., np.newaxis]

        mask = self.mask_source[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source[idx, :]
        label = torch.Tensor(np.array(label))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask}
        return sample


class HealthMNISTDataset(Dataset):
    """
    Dataset definition for the Health MNIST dataset when using simple MLP-based VAE.

    Data formatted as dataset_length x 1296.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, x_order, surv_x, cut_time, transform=None):

        self.data_source, label_df, self.mask_source = dataset_cut(csv_file_data, csv_file_label, mask_file,
                                                                   root_dir, cut_time)

        # self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        # self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        # label_df = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.label_source = label_df.iloc[:,x_order]
        # if surv_x:
        self.surv_source = label_df.iloc[:,surv_x]
        # else:
        #     self.surv_source = None
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array([digit], dtype='uint8')

        mask = self.mask_source.iloc[idx, :]
        mask = np.array([mask], dtype='uint8')

        label = self.label_source.iloc[idx, :]
        # changed
        # time_age,  disease_time,  subject,  gender,  disease,  location
        # label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))
        # label = torch.Tensor(np.nan_to_num(np.array(label)))
        # label = torch.Tensor(np.nan_to_num(np.array(label.iloc[[0,1,4,7,3,2]])))
        if self.surv_source is not None:
            surv_covariates = self.surv_source.iloc[idx, :]
            surv_covariates = torch.Tensor(np.nan_to_num(np.array(surv_covariates)))
        else:
            surv_covariates = None
        # subject, start_obs, gender, stop_age, event, stop_obs
        label = torch.Tensor(np.nan_to_num(np.array(label)))

        if self.transform:
            digit = self.transform(digit).reshape((-1))

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'surv_covariates':surv_covariates}
        return sample


class HealthMNISTDatasetConv(Dataset):
    """
    Dataset definiton for the Health MNIST dataset when using CNN-based VAE.

    Data formatted as dataset_length x 36 x 36.
    """

    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, x_order, surv_x, cut_time, id_covariate, transform=None):

        self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
        label_df = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.label_source = label_df.iloc[:,x_order]
        self.surv_source = label_df.iloc[:,surv_x]

        self.root_dir = root_dir
        self.transform = transform
        self.max_seq_length = label_df.groupby(label_df.columns[id_covariate]).count().max()[0]

        # num_bins = 80  # Example number of bins
        #
        # # 1. Find min and max
        # min_timestamp = self.label_source['start_obs'].min()
        # max_timestamp = self.label_source['start_obs'].max()
        # max_timestamp = max(max_timestamp, 40)
        # # 2. Create bin edges
        # bin_edges = np.linspace(min_timestamp, max_timestamp, num_bins)
        #
        # # Function to map each timestamp to the start of its bin
        # def map_to_bin_start(timestamp):
        #     # Find the index of the bin this timestamp belongs to
        #     bin_index = np.digitize(timestamp, bin_edges) - 1
        #     # Return the start of the bin
        #     return bin_edges[max(bin_index, 0)]
        # # 3. Assign data to bins
        # self.label_source['bins'] = self.label_source['start_obs'].apply(map_to_bin_start)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of bounds")
            return self.get_item(key)
        else:
            raise TypeError

    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        digit = self.data_source.iloc[idx, :]
        digit = np.array(digit, dtype='uint8')
        digit = digit.reshape(36, 36)
        digit = digit[..., np.newaxis]
        mask = self.mask_source.iloc[idx, :]
        label = self.label_source.iloc[idx, :]

        mask = np.array([mask], dtype='uint8')

        # CHANGED
        # label = torch.Tensor(np.nan_to_num(np.array(label[np.array([6, 4, 0, 5, 3, 7])])))
        # label = torch.Tensor(np.nan_to_num(np.array(label[np.array([0,1,2,3,4,5,8])])))
        # label = torch.Tensor(np.nan_to_num(np.array(label[np.array([0,1,4,7,3,2])])))
        # label = torch.Tensor(np.nan_to_num(np.array(label.iloc[0,1,4,7,3,2])))
        # subject, start_obs, gender, stop_age, event, stop_obs
        # label = torch.Tensor(np.nan_to_num(np.array(label.iloc[[0,1,4,7,3,2]])))

        if self.surv_source is not None:
            surv_covariates = self.surv_source.iloc[idx, :]
            surv_covariates = torch.Tensor(np.nan_to_num(np.array(surv_covariates)))
        else:
            surv_covariates = None
        # subject, start_obs, gender, stop_age, event, stop_obs
        label = torch.Tensor(np.nan_to_num(np.array(label)))

        if self.transform:
            digit = self.transform(digit)

        sample = {'digit': digit, 'label': label, 'idx': idx, 'mask': mask, 'surv_covariates':surv_covariates}
        return sample


def dataset_cut(csv_file_data, csv_file_label, mask_file, root_dir, cut_time):
    data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
    mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None)
    label_df = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)

    id_col = 'subject'
    time_col = 'start_obs'

    # Determine the maximum time point for each id in label_df
    max_time_by_id = label_df.groupby(id_col)[time_col].max()

    # Calculate the cut-off time for each id
    cut_off_time_by_id = max_time_by_id - cut_time

    # Merge the cut-off times with the label_df on 'id' to associate each row with its cut-off time
    label_df_with_cutoff = label_df.merge(cut_off_time_by_id.rename('cut_off_time'), on=id_col, how='left')

    # Filter out rows where the 'time' is greater than the 'cut_off_time'
    kept_rows_label_df = label_df_with_cutoff[label_df_with_cutoff[time_col] <= label_df_with_cutoff['cut_off_time']]

    # Step 1: Identify the new last observations for each id after cutting
    new_last_obs = kept_rows_label_df.sort_values(by=[id_col, time_col]).drop_duplicates(id_col, keep='last')

    # Step 2: For each id in new_last_obs, update the 'event' information
    # Note: This assumes that the original last observation's 'event' should be applied to the new last observation

    # Create a dictionary mapping id to the event of its original last observation
    event_mapping = label_df.sort_values(by=[id_col, time_col]).drop_duplicates(id_col, keep='last').set_index(id_col)[
        'event'].to_dict()

    # Update the 'event' column of the new last observations using the event_mapping
    new_last_obs['event'] = new_last_obs[id_col].map(event_mapping)

    # Step 3: Update kept_rows_label_df with the new event information for the new last observations
    # First, set the event to NaN for the old new last observations
    kept_rows_label_df.loc[new_last_obs.index, 'event'] = np.nan

    # Then, update the event information from new_last_obs
    kept_rows_label_df.update(new_last_obs[[id_col,time_col, 'event']])

    # Now, kept_rows_label_df contains the updated event information for the new last observations

    # Get the indexes of the kept rows
    kept_indexes = kept_rows_label_df.index

    # Filter data_source and mask_source by the kept indexes
    data_source = data_source.loc[kept_indexes]
    mask_source = mask_source.loc[kept_indexes]
    label_df = kept_rows_label_df.drop("cut_off_time", axis=1)

    return data_source, label_df, mask_source


class HUSCorogeneDataset(Dataset):
    """
    Dataset definiton for the HUS Corogene dataset when using FCN-based L-VAE.

    Data formatted as dataset_length x D.
    """
    def __init__(self, csv_file_data, csv_file_label, mask_file, root_dir, x_order, surv_x, cut_time, normalization_mode='standard',
                 mean_min=None, std_max=None):

        self.data_source, label_df, self.mask_source = dataset_cut(csv_file_data, csv_file_label, mask_file,
                                                                   root_dir, cut_time)
        # self.data_source = pd.read_csv(os.path.join(root_dir, csv_file_data), header=None)
        # self.mask_source = pd.read_csv(os.path.join(root_dir, mask_file), header=None).iloc[:,:100]
        if mean_min is not None and std_max is not None and normalization_mode == 'standard':
            self.data_source = (self.data_source - mean_min) / std_max
        elif mean_min is not None and std_max is not None and normalization_mode == 'normal':
            self.data_source = (self.data_source - mean_min) / (std_max - mean_min)
        else:
            data = self.data_source.copy().values
            data, mean_min, std_max = get_Normalization(data, self.mask_source.copy().values, normalization_mode)
            self.data_source = pd.DataFrame(data, columns=self.data_source.columns)

        self.std_max = std_max
        self.mean_min = mean_min
        ##TODO[HUS] : mask file usage to be corrected
        # label_df = pd.read_csv(os.path.join(root_dir, csv_file_label), header=0)
        self.label_source = label_df.iloc[:,x_order]
        self.surv_source = label_df.iloc[:,surv_x]
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_item(i) for i in range(start, stop, step)]
        elif isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of bounds")
            return self.get_item(key)
        else:
            raise TypeError


    def get_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data_source.iloc[idx, :]
        data = np.array(data)
        # data = data[..., np.newaxis]
        mask = self.mask_source.iloc[idx, :]
        label = self.label_source.iloc[idx, :]
        surv_covariates = self.surv_source.iloc[idx, :]

        mask = np.array([mask], dtype='uint8')

        surv_covariates = torch.Tensor(np.nan_to_num(np.array(surv_covariates)))
        # subject, start_obs, gender, stop_age, event, stop_obs
        label = torch.Tensor(np.nan_to_num(np.array(label)))


        sample = {'digit': data, 'label': label, 'idx': idx, 'mask': mask, 'surv_covariates':surv_covariates}
        return sample