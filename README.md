# CLOINet: Ocean State Reconstruction through Remote Sensing, In-Situ Observations, and Deep Learning

This repository contains the implementation of CLOINet (CLuster Optimal Interpolation Neural Network), a deep learning framework designed to reconstruct three-dimensional ocean states by integrating remote sensing data with sparse in-situ observations. This approach is detailed in our publication:

> **CLOINet: Ocean State Reconstructions through Remote-Sensing, In-Situ Sparse Observations and Deep Learning**  
> *Frontiers in Marine Science*, 2024.  
> [https://doi.org/10.3389/fmars.2024.1151868](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2024.1151868/full)

## Overview

CLOINet enhances traditional Optimal Interpolation (OI) methods by incorporating a self-supervised clustering mechanism. This allows for the segmentation of remote sensing images into clusters, revealing non-local correlations and improving the reconstruction of fine-scale oceanic features. The network is trained using outputs from an Ocean General Circulation Model (OGCM) and has demonstrated significant improvements over baseline OI techniques, including up to a 40% reduction in reconstruction error and the ability to resolve scales 50% smaller.

## Repository Structure

- **`CLOINet.py`**: Implements the main CLOINet architecture, combining OI principles with deep learning-based clustering.
- **`data_processing.py`**: Contains functions for preprocessing remote sensing and in-situ observation data.
- **`training_script.py`**: Script to train CLOINet using simulated or real-world datasets.
- **`evaluation.py`**: Tools for assessing the performance of the trained model against baseline methods.

## Getting Started

1. **Installation**: Clone the repository and install the required dependencies listed in `requirements.txt`.

## Citation

If you utilize this code in your research, please cite our paper:

```
@article{Cutolo2024CLOINet,
  title={CLOINet: Ocean State Reconstructions through Remote-Sensing, In-Situ Sparse Observations and Deep Learning},
  author={Cutolo, Eugenio and Pascual, Ananda and Ruiz, Simon and Zarokanellos, Nikolaos D. and Fablet, Ronan},
  journal={Frontiers in Marine Science},
  volume={11},
  year={2024},
  doi={10.3389/fmars.2024.1151868}
}
```

For detailed information on the methodology and experiments, refer to the full paper linked above. 

## License

This repository is released under the **GNU General Public License v3 (GPLv3)**, which ensures that the code remains **open-source** and that any modifications or derivative works must also be **shared under the same license**.  

By using or modifying this software, you **must**:  
✔ **Keep your modifications open-source** under **GPLv3**.  
✔ **Provide attribution** by citing the original paper and repository.

For full details, see the **[LICENSE.txt](LICENSE.txt)** file.
