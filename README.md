# DISYRE: Diffusion-Inspired SYnthetic Restoration for Unsupervised Anomaly Detection

Official implementation of _**DISYRE**_ and _**DISYRE v2**_.
## Installation



```bash
git clone https://github.com/snaval/disyre.git
cd disyre
```


## Data download and preprocessing

1) We use the Brain MRI preprocessing pipeline from the [UPD Study](https://github.com/iolag/UPD_study/tree/main). Follow the link to download the data and preprocess it. 
2) Modify `environment_defaults.py` to point to the correct data folders.
3) The specific files used for training and validation are specified in the _json_ files
that can be found in the `experiments` folder. These should be already setup in  `environment_defaults.py`
4) Random shapes are used for anomaly generation. The shapes that we used are in `experiments/shapes/` folder.

## Model training

1) Setup accelerate (we used `bf16` in our latest experiments): 

```bash
accelerate config
```


2) Experiment configuration

Training datasets (with synthetically generated anomalies) and testing datasets configurations are stored in the `experiments/dconf` folder.

```
experiments/dconf/
    bratst2.json
    dag2D_camcant2.json
    fpi2D_camcant2.json
...
```

Experiments are configured in the `experiments/002D` folder.

```
experiments/002D/
    dag01_t2.json
    fpi01_t2.json
    ...
```


3) Train the models with _DAG_ configuration:

```bash
accelerate launch trainer_disyre.py --json "experiments/002D/dag01_t1.json"
accelerate launch trainer_disyre.py --json "experiments/002D/dag01_t2.json"
```

## Model evaluation

After our MICCAI'24 paper we re-implemented the synthetic anomaly generation process and augmentations using the amazing 
[batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) library. 
The current implementation achieves slightly improved results in the ATLAS Brain MRI dataset.

While training, models are evaluated using only randomly sampled patches of the test set (i.e. without sliding window methodology). 
Full evaluation and visualizations can be run using the jupyter notebook `notebooks/evaluation.ipynb`.



## Citing 

If you use this implementation or build on our methods, please cite our papers:

 - [DISYRE: Diffusion-Inspired SYnthetic REstoration for Unsupervised Anomaly Detection](https://arxiv.org/abs/2311.15453) (ISBI'24):
    
```bibtex
@inproceedings{disyre,
  title={DISYRE: Diffusion-Inspired SYnthetic REstoration for Unsupervised Anomaly Detection},
  author={Naval Marimont, Sergio and Baugh, Matthew and Siomos, Vasilis and Tzelepis, Christos and Kainz, Bernhard and Tarroni, Giacomo},
  booktitle={Proceedings/IEEE International Symposium on Biomedical Imaging: from nano to macro. IEEE International Symposium on Biomedical Imaging},
  year={2024},
  organization={IEEE}
}
```

 - [Ensembled Cold-Diffusion Restorations for Unsupervised Anomaly Detection](https://arxiv.org/abs/2407.06635) (MICCAI'24):
    
```bibtex
@inproceedings{disyre_v2,
  title={Ensembled Cold-Diffusion Restorations for Unsupervised Anomaly Detection},
  author={Naval Marimont, Sergio and Baugh, Matthew and Siomos, Vasilis and Tzelepis, Christos and Kainz, Bernhard and Tarroni, Giacomo},
  booktitle={Proceedings/Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2024},
  organization={Springer}
}

```
    