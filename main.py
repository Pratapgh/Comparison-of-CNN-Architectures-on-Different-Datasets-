
# CNN_project/
# ├── data/                     # Directory for datasets
# ├── models/                   # Directory for CNN model architectures
# │   ├── lenet.py              # LeNet-5 implementation
# │   ├── alexnet.py            # AlexNet implementation
# │   ├── googlenet.py          # GoogLeNet implementation
# │   ├── vggnet.py             # VGGNet implementation
# │   ├── resnet.py             # ResNet implementation
# │   ├── exception.py          # Xception implementation
# │   ├── senet.py              # SENet implementation
# ├── tests/                    # Directory for model testing scripts
# │   ├── test_lenet.py         # Test script for LeNet-5
# │   ├── test_alexnet.py       # Test script for AlexNet
# │   ├── test_googlenet.py     # Test script for GoogLeNet
# │   ├── test_vggnet.py        # Test script for VGGNet
# │   ├── test_resnet.py        # Test script for ResNet
# │   ├── test_exception.py      # Test script for Xception
# │   ├── test_senet.py         # Test script for SENet
# │   └── run_tests.py          # Script to run all tests together
# ├── notebooks/                # (Optional) For Jupyter Notebooks
# ├── plots/                    # Save plots and graphs here
# ├── utils/                    # Directory for utility functions
# │   ├── data_loader.py        # Dataset loading and preprocessing
# │   ├── metrics.py            # Functions for evaluation metrics
# │   ├── visualization.py      # Functions for plotting results
# ├── main.py                   # Main script to run the project
# ├── requirements.txt          # Dependencies for the project
# ├── README.md                 # Project documentation
# ├── LICENSE                   # (Optional) License file
# └── .gitignore                # Git ignore file
#

#
# CNN_project/
# ├── data/
# │   ├── MNIST/
# │   │   ├── raw/
# │   │   │   ├── train-images.idx3-ubyte
# │   │   │   ├── train-labels.idx1-ubyte
# │   │   │   ├── t10k-images.idx3-ubyte
# │   │   │   ├── t10k-labels.idx1-ubyte
# │   ├── FMNIST/
# │   │   ├── raw/
# │   │   │   ├── train-images-idx3-ubyte
# │   │   │   ├── train-labels-idx1-ubyte
# │   │   │   ├── t10k-images-idx3-ubyte
# │   │   │   ├── t10k-labels-idx1-ubyte
# │   ├── CIFAR-10/
# │   │   ├── raw/
# │   │   │   ├── cifar-10-batches-bin/
# │   │   │   ├── cifar-10-python.tar.gz
