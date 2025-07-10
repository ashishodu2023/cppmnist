#pragma once
#include <torch/torch.h>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/samplers/random.h> 

#include "/home/ashishverma/Documents/cppminist/src/include/evaluate.hpp"
#include <iostream>
#include <iomanip>
#include "net.hpp"


template <typename DataLoader>
void evaluate(Net& model, DataLoader& loader, torch::Device device);


// Add below the declaration in evaluate.hpp
template <typename DataLoader>
void evaluate(Net& model, DataLoader& loader, torch::Device device) {
    model.eval();
    size_t correct = 0, total = 0;

    for (const auto& batch : loader) {
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);
        auto output = model.forward(data);
        auto pred = output.argmax(1);
        //correct += pred.eq(targets).sum().item<int64_t>();
        correct += pred.eq(targets).sum().template item<int64_t>();
        total += targets.size(0);
    }

    float acc = static_cast<float>(correct) / total * 100.0f;
    std::cout << "Test Accuracy: " << std::setprecision(4) << acc << "%\n";
}