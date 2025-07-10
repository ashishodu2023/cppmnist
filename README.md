
#  MNIST Digit Classifier (C++ / LibTorch)

This project implements a handwritten digit classifier using the **MNIST** dataset, built with **LibTorch** (the C++ API for PyTorch) and accelerated by **CUDA** (if available).

---

##  Features

- ✅ Written in modern **C++17**
- ✅ Uses **LibTorch** (PyTorch C++ frontend)
- ✅ **CUDA GPU acceleration** (automatically used if available)
- ✅ MNIST training + test evaluation
- ✅ Modular code (`Net`, `Evaluate`, `main`)
- ✅ Model saving & loading (`torch::save` / `torch::load`)

---

##  Project Structure

cppminist/
├── CMakeLists.txt
├── models/
│ └── mnist_model.pt # Saved model
├── data/ # MNIST data directory
├── src/
│ ├── main.cpp # Training & evaluation entrypoint
│ ├── net.cpp # Model architecture implementation
│ ├── evaluate.cpp # Evaluation loop
│ └── include/
│ ├── net.hpp # Model header
│ └── evaluate.hpp # Eval header



---

##  Requirements

- **CMake 3.15+**
- **g++ 9+** with C++17 support
- **CUDA Toolkit** (optional but recommended)
- **LibTorch** (download from [pytorch.org](https://pytorch.org/get-started/locally/#libtorch))

---

## 🛠️ Build & Run

### 1. Clone this repo

```bash
git clone https://github.com/yourname/cppminist.git
cd cppminist

# CUDA version
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-*.zip


mkdir -p ./data
cd ./data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
cd ..


mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=../libtorch ..
make

./mnist_classifier
```

```ymal

✅ Using device: GPU
Epoch: 1 [57664/60000] Loss: 0.291559
🧪 Test Accuracy: 93.84%
💾 Model saved to ./models/mnist_model.pt
📂 Loaded model, evaluating:
🧪 Test Accuracy: 93.84%

```

```cpp
Net() {
  conv1 = register_module("conv1", torch::nn::Conv2d(1, 10, 5));
  conv2 = register_module("conv2", torch::nn::Conv2d(10, 20, 5));
  fc1 = register_module("fc1", torch::nn::Linear(320, 50));
  fc2 = register_module("fc2", torch::nn::Linear(50, 10));
}
```


### TODO 

* Add training/validation loss graph (using matplotlib-cpp or file output)
* Export model as TorchScript
* Add CLI support for inference
* Support additional datasets (e.g., CIFAR-10)



### CREDITS 

    Built by Ashish Verma

    Inspired by PyTorch’s C++ frontend examples

    MNIST dataset from Yann LeCun’s site