import json
import os
import nibabel as nib
from PIL import Image
from torchvision import transforms as T

import numpy as np
import json

from argparse import Namespace
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.abstract_transforms import AbstractTransform

def load_datalist(file_path, key='training', base_dir=None, meta_keys=[]):
    with open(file_path, 'r') as f:
        datalist = json.load(f)[key]
    if base_dir is None:
        base_dir = os.path.dirname(file_path)
    datalist = [{k:os.path.join(base_dir, v) if k not in meta_keys else v for k,v in i.items()} for i in datalist]
    return datalist


class ScaleIntensityQuantile:
    def __init__(self, min_p, max_p):
        self.min_p = min_p
        self.max_p = max_p

    def __call__(self, **data_dict):
        for k,v in data_dict.items():
            v = np.clip(v, 0, None) # Remove -ve values
            minmax_q = np.quantile( v.flatten(), q= np.array((self.min_p,self.max_p)))
            v -= minmax_q[0]
            v /= (minmax_q[1] - minmax_q[0])
            v = np.clip(v, 0, 1)
            data_dict[k] = v
        return data_dict


class CropForeground(AbstractTransform):
    def __init__(self, key_input = "data", keys_to_apply = None):
        """
        Crop the foreground of an image
        :param keys_inputs:
        :param keys_to_apply:
        """
        self.key_input = key_input
        self.keys_to_apply = [key_input] if keys_to_apply is None else (keys_to_apply if isinstance(keys_to_apply, (list, tuple)) else [keys_to_apply])

    def __call__(self, **data_dict):

        outputs = {k:[] for k in self.keys_to_apply}
        for i, (data,metadata) in enumerate(zip(data_dict[self.key_input],data_dict["metadata"])):

            nonzero = np.stack(np.abs(data).sum(0).nonzero(),-1) # Reduce channel dimension and get coords not zero

            if nonzero.shape[0] != 0:
                nonzero = np.stack([np.min(nonzero, 0), np.max(nonzero, 0)],-1)
                # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis
                for key in self.keys_to_apply:
                    if key in data_dict:
                        seg = data_dict[key][i]
                        if seg is not None:
                            # now crop to nonzero
                            seg = seg[:,
                                       nonzero[0, 0]: nonzero[0, 1] + 1,
                                       nonzero[1, 0]: nonzero[1, 1] + 1,
                                       ]
                            if nonzero.shape[0] == 3:
                                seg = seg[:,:,:, nonzero[2, 0]: nonzero[2, 1] + 1]

                            outputs[key].append(seg)

                metadata["nonzero_region"] = nonzero
            else:
                for key in self.keys_to_apply:
                    outputs[key].append(data_dict[key][i])

        # Note that the output of this is a list, instead of single numpy array because each cna have different sizes
        # Hope for anything that comes after to iterate over the list and does not expect a np.array
        for key in self.keys_to_apply:
            data_dict[key] = outputs[key]

        return data_dict

def default(config:Namespace, key, default):
    if hasattr(config,key):
        return getattr(config,key)
    return default

def load_dataset_config(file_path):

    import environment_defaults

    config = json.load(open(file_path, 'r'))

    assert config['dataset_type'] in ["training", "evaluation", "evaluation_image"]
    assert "dataset_name" in config

    # Load default values from environment_defaults
    from_environment = [(f'{config["dataset_name"]}_json',True),(f'{config["dataset_name"]}_base_dir',False)]

    if config['dataset_type'] == "training":
        from_environment.extend([('shape_dataset',True), ('shape_base_dir',False)])

    for k,required in from_environment:
        if k not in config:
            if k in os.environ:
                config[k] = os.environ[k]

        assert (not required) or (k in config) or (k in os.environ),f"Missing setting {k}, not found in config or environment variables"

    # Transform config dict to Namespace
    config = Namespace(**config)

    return config

class DataLoader3D(DataLoader):
    def __init__(self, datalist_json, datalist_key,  keys_image, batch_size, num_threads_in_multithreaded=0,
                 keys_label = None, datalist_base_dir=None, seed_for_shuffle=1234,
                 return_incomplete=True, shuffle=True, infinite=True, intensity_percentile_scaling = None,
                 meta_keys= []):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        meta_keys = meta_keys if meta_keys is not None else []

        self.datalist = load_datalist(datalist_json,datalist_key,datalist_base_dir, meta_keys)
        super().__init__(self.datalist, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)

        self.keys_image = keys_image if isinstance(keys_image,(tuple,dict) ) else [keys_image]
        if keys_label:
            self.keys_label = keys_label if isinstance(keys_label, (tuple,dict)) else [keys_label]
        else:
            self.keys_label = None
        self.meta_keys = meta_keys
        self.indices = list(range(len(self.datalist)))

        if intensity_percentile_scaling is not None:
            self.intensity_scaling = ScaleIntensityQuantile(*intensity_percentile_scaling)
        else:
            self.intensity_scaling = None

    @staticmethod
    def load_file(file):
        data = nib.load(file)
        data_array = np.asanyarray(data.dataobj, order="C").astype(np.float32)
        # data_array = data.get_fdata()
        return data_array, {"affine": data.affine,"filename":file}

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []
        if self.keys_label:
            labels = []

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            for j, k in enumerate(self.keys_image):
                if k in subj:
                    img, img_mtd = self.load_file(subj[k])
                    if self.intensity_scaling is not None:
                        img = self.intensity_scaling(data=img)['data']
                    patient_data.append(img)
                    if j == 0:
                        for k in self.meta_keys:
                            img_mtd[k] = subj[k]

                        metadata.append(img_mtd)
                        patient_names.append(subj[k])

            images.append(np.stack(patient_data,axis=0))

            # this will only pad patient_data if its shape is smaller than self.patch_size
            if self.keys_label:
                patient_label = []
                for j, k in enumerate(self.keys_label):
                    if k in subj:
                        seg, _ = self.load_file(subj[k])
                        patient_label.append(seg)
                patient_label = np.stack(patient_label, axis=0)
                labels.append(patient_label)

        if self.keys_label:
            return {'data': images, 'seg': labels, 'metadata': metadata, 'names': patient_names}
        else:
            return {'data': images, 'metadata': metadata, 'names': patient_names}


class DataLoader2Dfrom3D(DataLoader):
    def __init__(self, datalist_json, datalist_key, keys_image, batch_size, num_threads_in_multithreaded=0,
                 keys_label=None, datalist_base_dir=None, seed_for_shuffle=1234, num_slices=None,
                 return_incomplete=True, shuffle=True, infinite=True, intensity_percentile_scaling = None,
                 meta_keys=[]):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        """
        meta_keys = meta_keys if meta_keys is not None else []

        self.datalist = load_datalist(datalist_json, datalist_key, datalist_base_dir, meta_keys)
        super().__init__(self.datalist, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete,
                         shuffle,
                         infinite)

        self.keys_image = keys_image if isinstance(keys_image, (tuple, dict)) else [keys_image]
        if keys_label:
            self.keys_label = keys_label if isinstance(keys_label, (tuple, dict)) else [keys_label]
        else:
            self.keys_label = None
        self.meta_keys = meta_keys
        self.indices = list(range(len(self.datalist)))
        self.num_slices = num_slices

        if intensity_percentile_scaling is not None:
            self.intensity_scaling = ScaleIntensityQuantile(*intensity_percentile_scaling)
        else:
            self.intensity_scaling = None

    @staticmethod
    def load_file(file):
        return DataLoader3D.load_file(file)

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []
        if self.keys_label:
            labels = []

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            for j, k in enumerate(self.keys_image):
                if k in subj:
                    img, img_mtd = self.load_file(subj[k])
                    if self.intensity_scaling is not None:
                        img = self.intensity_scaling(data=img)['data']

                    patient_data.append(img)
                    if j == 0:
                        for k in self.meta_keys:
                            img_mtd[k] = subj[k]
                        num_slices = self.num_slices if self.num_slices is not None else img.shape[-1]
                        metadata.extend([img_mtd,]*num_slices)
                        patient_names.extend([subj[k],]*num_slices)

            patient_data = np.stack(patient_data, axis=0)

            # Get non-empty axial slices
            list_ax_sl_ind = np.abs(patient_data).reshape(-1, patient_data.shape[-1]).sum(0)
            list_ax_sl_ind = list_ax_sl_ind.nonzero()[0]

            if self.num_slices is not None:
                # Randomly sample num_slices slices
                z_slices = np.random.randint(list_ax_sl_ind.min(),list_ax_sl_ind.max(),size=(self.num_slices,))
                images.extend(list(np.moveaxis(patient_data[:,:,:,z_slices],-1,0)))
            else:
                # Get all non-empty slices
                images.extend(list(np.moveaxis(patient_data[:,:,:,list_ax_sl_ind], -1, 0)))

            # this will only pad patient_data if its shape is smaller than self.patch_size
            if self.keys_label:
                patient_label = []
                for j, k in enumerate(self.keys_label):
                    if k in subj:
                        seg, _ = self.load_file(subj[k])
                        patient_label.append(seg)
                patient_label = np.stack(patient_label, axis=0)
                if self.num_slices is not None:
                    labels.extend(list(np.moveaxis(patient_label[:,:,:,z_slices],-1,0)))
                else:
                    labels.extend(list(np.moveaxis(patient_label[:,:,:,list_ax_sl_ind], -1, 0)))

        if self.keys_label:
            return {'data': images, 'seg': labels, 'metadata': metadata, 'names': patient_names}
        else:
            return {'data': images, 'metadata': metadata, 'names': patient_names}


def nibabel_loader(file):
        data = nib.load(file)
        data_array = np.asanyarray(data.dataobj, order="C").astype(np.float32)
        # data_array = data.get_fdata()
        return data_array, {"affine": data.affine,"filename":file}


class DataLoader2D(DataLoader):
    def __init__(self, datalist_json, datalist_key,  keys_image, batch_size, num_threads_in_multithreaded=0,
                 keys_label = None, datalist_base_dir=None, seed_for_shuffle=1234,
                 return_incomplete=True, shuffle=True, infinite=True,
                 intensity_percentile_scaling = None,
                 image_loader="nibabel", preprocessing_image = None, meta_keys=[] ):
        """
        data must be a list of patients as returned by get_list_of_patients (and split by get_split_deterministic)

        patch_size is the spatial size the retured batch will have

        image_loader: ["nibabel",]

        """
        meta_keys = meta_keys if meta_keys is not None else []
        self.datalist = load_datalist(datalist_json,datalist_key,datalist_base_dir, meta_keys)
        super().__init__(self.datalist, batch_size, num_threads_in_multithreaded, seed_for_shuffle, return_incomplete, shuffle,
                         infinite)

        self.keys_image = keys_image if isinstance(keys_image,(tuple,dict) ) else [keys_image]
        if keys_label:
            self.keys_label = keys_label if isinstance(keys_label, (tuple,dict)) else [keys_label]
        else:
            self.keys_label = None
        self.meta_keys = meta_keys
        self.indices = list(range(len(self.datalist)))

        if intensity_percentile_scaling:
            self.intensity_scaling = ScaleIntensityQuantile(*intensity_percentile_scaling)
        else:
            self.intensity_scaling = None

        if image_loader == "nibabel":
            self.loader = nibabel_loader
        else:
            raise ValueError(f"Data type {image_loader} not supported, options supported ['nibabel']")

        self.preprocessing_image = preprocessing_image

    def load_file(self,file):
        return self.loader(file)

    def generate_train_batch(self):
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        idx = self.get_indices()
        patients_for_batch = [self._data[i] for i in idx]

        # initialize empty array for image and labels
        images = []
        if self.keys_label:
            labels = []

        metadata = []
        patient_names = []

        # iterate over patients_for_batch and include them in the batch
        for i, subj in enumerate(patients_for_batch):
            patient_data = []
            for j, k in enumerate(self.keys_image):
                if k in subj:
                    img, img_mtd = self.load_file(subj[k])
                    if self.intensity_scaling is not None:
                        img = self.intensity_scaling(data=img)['data']
                    patient_data.append(img)
                    if j == 0:
                        for k in self.meta_keys:
                            img_mtd[k] = subj[k]
                        metadata.append(img_mtd)
                        patient_names.append(subj[k])

            images.append(np.stack(patient_data,axis=0))

            # this will only pad patient_data if its shape is smaller than self.patch_size
            if self.keys_label:
                patient_label = []
                for j, k in enumerate(self.keys_label):
                    if k in subj:
                        seg, _ = self.load_file(subj[k])
                        patient_label.append(seg)
                patient_label = np.stack(patient_label, axis=0)
                labels.append(patient_label)

        if self.keys_label:
            return {'data': np.stack(images), 'seg': np.stack(labels), 'metadata': metadata, 'names': patient_names}
        else:
            return {'data': np.stack(images), 'metadata': metadata, 'names': patient_names}