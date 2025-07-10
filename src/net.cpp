#include "../include/net.hpp"

Net::Net() {
    conv1 = register_module("conv1", torch::nn::Conv2d(1, 10, 5));
    conv2 = register_module("conv2", torch::nn::Conv2d(10, 20, 5));
    fc1   = register_module("fc1", torch::nn::Linear(320, 50));
    fc2   = register_module("fc2", torch::nn::Linear(50, 10));
}

torch::Tensor Net::forward(torch::Tensor x) {
    x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
    x = torch::relu(torch::max_pool2d(conv2->forward(x), 2));
    x = x.view({-1, 320});
    x = torch::relu(fc1->forward(x));
    x = torch::log_softmax(fc2->forward(x), 1);
    return x;
}

torch::nn::Conv2d& Net::get_conv1() { return conv1; }
torch::nn::Conv2d& Net::get_conv2() { return conv2; }
torch::nn::Linear& Net::get_fc1()   { return fc1; }
torch::nn::Linear& Net::get_fc2()   { return fc2; }

void Net::save(torch::serialize::OutputArchive& archive) const {
    torch::nn::Module::save(archive); // saves all registered modules
}

void Net::load(torch::serialize::InputArchive& archive) {
    torch::nn::Module::load(archive); // loads all registered modules
}
