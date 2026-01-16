# COrAL: Orthogonalized Multimodal Contrastive Learning with Asymmetric Masking 
COrAL, a pioneering framework that explicitly and simultaneously preserves all three multimodal information components (redundant, unique, and synergistic), features a dual-path architecture with orthogonality constraints to enforce a clean separation between shared and unique representations, combined with asymmetric masking using complementary view-specific patterns, which encourages cross-modal inference for synergy capture.
## How to run COrAL
### Installation
You can clone the code and install all required packages by running the following lines:
```
git clone https://github.com/Caro-Ci/COrAL && cd Coral
conda env create -f environment.yml
conda activate coral
```
### Experiments on MultiBench data
We evaluate COrAL on the 5 MultiBench datasets CMU-MOSI, CMU-MOSEI, MIMIC III, UR-FUNNY and MUsTARD. For MIMIC III, data is only available if you have the necessary credentials. Before running the script, please adjust the paths in `dataset/catalog.json` accordingly. This will run COrAL with five different seeds for the chosen dataset:
```
pyhon main_multibench.py --dataset="mosi" #Can be "mosi", "mosei", "mimic", "humor" or "sarcasm"
```
### Experiments on synthetic trifeature dataset
To evaluate the capture of all three information components with COrAL, we use the trifeature dataset. When first running it, the dataset will be generated first. This will run COrAL with five different seeds and give accuracies for three downstream tasks that depend on one information component each:
```
pyhon main_trifeature.py 
```
## Thanks
We wish to thank the contributors of the repository [CoMM](https://github.com/Duplums/CoMM), which has greatly aided the implementation of COrAL.
