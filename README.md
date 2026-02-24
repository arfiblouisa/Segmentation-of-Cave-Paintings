# Segmentation of Cave Paintings

## Overview

<!-- This repository contains the code and resources associated with the article:

> **[Title of Your Paper]**  
> *Authors*, Year. -->

This project focuses on the automatic segmentation of cave paintings from 2D images.  
The primary objective is to extract painted regions (foreground) from complex backgrounds affected by uneven illumination, rock texture, and pigment degradation.


## General Pipeline

The overall processing pipeline is illustrated below:

![General Pipeline](general_pipeline.png)

The segmentation workflow consists of the following main steps:

1. **Human cropping** around the area of interest (typically a specific painting)
2. **Preprocessing:** color space conversion, contrast enhancement (via CLAHE) and noise reduction.  
3. **Foreground extraction using a specific segmentation algorithm**
4. **Final binary mask generation**


## Aim of the Project

Cave paintings present several segmentation challenges:

- Irregular rock surfaces  
- Non-uniform illumination  
- Low contrast between pigments and background  
- Surface degradation and noise  

The goal of this project is to provide:

- A reproducible segmentation framework   
- A modular implementation suitable for heritage imaging applications  
- Baseline tools for further quantitative or morphological analysis  

<!-- This repository accompanies the published article and ensures transparency and reproducibility of the experiments. -->


## Repository Structure

Below is a description of the main files and directories:

- **dataset**
  49 cave paintings images that can be used to test the codes. All images can be used and cited using the *infos.txt* files to correctly cite the authors. 

- **src/seg_cave_paintings**
  All the utils functions used to segment the painting of interest. They are called in *demo_notebooks* and *full_interactive_pipeline*.

- **demo_notebooks**
  Jupyter notebooks to test the three methods described in the article : Otsu, active conctours and meanshift. 
  *get_rectangles_coordinates.py* permits the user to get the bounding box coordinates of the object they want to segment.

- **SAM**
  Repository to experiment with SAM3 model. A selection of pictures from *./dataset/* are available in *./SAM/prompt_dataset* to experiment, but other images can be used. 
  *./SAM/SAM3_prompt_experiments.ipynb* should be used to generate the segmentation masks.

- **preprocessing_comparisons**
  Comparisons of different preprocessing methods ono different types of cave paintings (charcoal, red, and both charcoal and red). The different results are presented under .png format.

---

## Installation

We advise you to use Conda for this project. If you intend to run the SAM notebook, you will need to have :
- Python 3.12 or higher
- PyTorch 2.7 or higher
- CUDA-compatible GPU with CUDA 12.6 or higher


1. **Create a new Conda environment:**

```bash
conda create -n seg_cave_env python=3.12
conda activate seg_cave_env
```

2. **Optional : Install PyTorch with CUDA support:**
If you don't intend to re-run the SAM notebook, pythorch isn't needed.
```bash
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

3. **Clone the repository and install the package:**

Clone our repository :
```bash
git clone https://github.com/arfiblouisa/Segmentation-of-Cave-Paintings
cd Segmentation-of-Cave-Paintings
pip install -r requirements.txt
```

Clone SAM3 repository :
```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```
Install additional SAM dependencies:
```bash
# For running example notebooks
pip install -e ".[notebooks]"
```

⚠️ Before using SAM 3, you need to request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication).

## Usage :
```bash
python main.py --input path/to/image.jpg --output results/
```

<!-- ## Citation :
@article{yourcitation,
  title={Title},
  author={Authors},
  journal={Journal},
  year={Year}
} -->

## Contact

For questions or collaborations, please contact:  
Louisa Arfib : louisa.arfib@student-cs.fr  
Manon Arfib : manon.arfib@student-cs.fr