# Artificial Intelligence Predicts Features Associated with Breast Cancer Neoadjuvant Chemotherapy Response from Multi-stain Histopathologic Images
Huang <em>et al.</em> 2022


[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/huangzhii/IMPRESS-Breast-Cancer/blob/master/LICENSE)


## IRB Information
This study was approved by the Ohio State University Institutional Research Board and included 62 HER2-positive breast cancer (HER2+) patients and 64 triple-negative breast cancer (TNBC) patients treated with neoadjuvant chemotherapy (NAC) and follow-up surgical excision. Patients with histopathologically confirmed invasive breast carcinoma who underwent NAC from January 2011 to December 2016, those who had underwent surgery after completing NAC were included.

## Workflow
We aimed to investigate whether artificial intelligence (AI)-based algorithms using automatic feature extraction methods can predict neoadjuvant chemotherapy (NAC) outcome in HER2-positive (HER2+) and triple-negative breast cancer (TNBC) patients. In this study, an automatic, accurate, comprehensive, interpretable, and reproducible whole slide image (WSI) feature extraction pipeline was constructed, IMage-based Pathological REgistration and Segmentation Statistics (IMPRESS). The developed machine learning models using image-based features derived from tumor and tumor microenvironment, biomarkers and clinical features accurately predicted the response to NAC in breast cancer patients, and outperformed the results learned by pathologists' manually assessed features.

<div style="text-align:center"><img src="figure/figure_flowchart_72dpi.png" width=1000/></div>

## Software Requirements
* pandas 1.0.5
* pytorch 1.6.0
* torchvision 0.7.0
* scikit-learn 0.23.2
* pillow 7.2.0
* OpenCV 4.4.0
* openslide 1.1.2


## Run Experiments

* Users can use main.py to run machine learning experiments.
```bash
python main.py
```
* After machine learning experiments, users can use summary.py to generate results.
```bash
python summary.py
```
* Alternatively, a bash file can help users to run all experiments & analyses altogether using 10 different random seeds:
```bash
bash autorun.sh
```


## Folder Explanation
- figure/

    This folder contains some displaying figures.
    
- features/

    This folder contains IMPRESS and pathologists' assessed features for HER2+ and TNBC cohorts.
    
- pipeline/

    This folder contains the entire IMPRESS pipeline.
    
- clinical/

    This folder contains clinical information for HER2+ and TNBC cohorts.



## Contact Information
Zhi Huang

hz9423@gmail.com
