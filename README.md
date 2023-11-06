# ECGTransForm: Empowering Adaptive ECG Arrhythmia Classification Framework with Bidirectional Transformer [[Paper](https://)] [[Cite](#citation)]
#### *by: Hany El-Ghaish, Emadeldeen Eldele*
#### This work is accepted for publication in the Biomedical Signal Processing and Control.

## Abstract
Cardiac arrhythmias, deviations from the normal rhythmic beating of the heart, are subtle yet critical indicators of potential cardiac challenges. Efficiently diagnosing them requires intricate understanding and representation of both spatial and temporal features present in Electrocardiogram (ECG) signals. This paper introduces ECGTransForm, a deep learning framework tailored for ECG arrhythmia classification. By embedding a novel Bidirectional Transformer (BiTrans) mechanism, our model comprehensively captures temporal dependencies from both antecedent and subsequent contexts. This is further augmented with Multi-scale Convolutions and a Channel Recalibration Module, ensuring a robust spatial feature extraction across various granularities. We also introduce a Context-Aware Loss (CAL) that addresses the class imbalance challenge inherent in ECG datasets by dynamically adjusting weights based on class representation. Extensive experiments reveal that ECGTransForm outperforms contemporary models, proving its efficacy in extracting meaningful features for arrhythmia diagnosis. Our work offers a significant step towards enhancing the accuracy and efficiency of automated ECG-based cardiac diagnoses, with potential implications for broader cardiac care applications.


## Datasets
We used two public datasets in this study (The preprocessed datasets are available [Here](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)):
- [MIT-BIH](https://www.physionet.org/content/mitdb/1.0.0/)
- [PTB](https://physionet.org/content/ptbdb/1.0.0/)

### Configurations
There are two configuration files: 
- one for dataset configuration `configs/data_configs.py`
- one for training configuration `configs/hparams.py`

### Citation:
If you found this work useful for you, please consider citing it.
```
@ARTICLE{ecgTransForm,
  
}
```

