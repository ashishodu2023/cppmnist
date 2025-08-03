#pragma once
#include <torch/torch.h>

class Net : public torch::nn::Module {
public:
    Net();

    torch::Tensor forward(torch::Tensor x);

    void save(torch::serialize::OutputArchive& archive) const override;
    void load(torch::serialize::InputArchive& archive) override;

private:
    torch::nn::Sequential features{nullptr}, classifier{nullptr};
};
