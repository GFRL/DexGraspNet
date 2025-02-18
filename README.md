# BODEX baseline: DexGraspNet

### BODex: [Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

### DexGraspNet: [[project page]](https://pku-epic.github.io/DexGraspNet/)

## Environment
swithch to the main branch and follow the instructions in the README.md file to install the required packages.

## Generate the dataset

1. Link the dataset to the assets folder
ln -s ${YOUR_DATA_PATH} assets/DGNObj
ln -s ${YOUR_SPLIT_PATH} assets/DGNObj_split

2. Generate the dataset
```
cd grasp_generation
python main.py # generate grasps for single object
python multi_plan.py        # generate grasps for multiple objects
