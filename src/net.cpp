#include "../include/net.hpp"

Net::Net() {
    features = register_module("features", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 5)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
        torch::nn::ReLU(),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 5)),
        torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2)),
        torch::nn::ReLU()
    ));

    classifier = register_module("classifier", torch::nn::Sequential(
        torch::nn::Linear(320, 50),
        torch::nn::ReLU(),
        torch::nn::Linear(50, 10),
        torch::nn::LogSoftmax(torch::nn::LogSoftmaxOptions(1))
    ));
}

torch::Tensor Net::forward(torch::Tensor x) {
    x = features->forward(x);
    x = x.view({-1, 320});
    x = classifier->forward(x);
    return x;
}

void Net::save(torch::serialize::OutputArchive& archive) const {
    torch::nn::Module::save(archive);
}

void Net::load(torch::serialize::InputArchive& archive) {
    torch::nn::Module::load(archive);
}
