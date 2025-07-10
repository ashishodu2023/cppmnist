#pragma once
#include <torch/torch.h>

class Net : public torch::nn::Module {
public:
    Net();

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d& get_conv1();
    torch::nn::Conv2d& get_conv2();
    torch::nn::Linear& get_fc1();
    torch::nn::Linear& get_fc2();

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
