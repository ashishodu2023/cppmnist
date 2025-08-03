#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include "/home/ashishverma/Documents/cppmnist/src/include/evaluate.hpp"
#include "/home/ashishverma/Documents/cppmnist/src/include/net.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

template <typename DataLoader>
void train(Net& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer, torch::Device device, std::vector<float>& loss_values, size_t epoch) {
    model.train();
    size_t batch_idx = 0;
    for (auto& batch : data_loader) {
        optimizer.zero_grad();
        auto data = batch.data.to(device);
        auto targets = batch.target.to(device);

        auto output = model.forward(data);
        auto loss = torch::nll_loss(output, targets);
        loss.backward();
        optimizer.step();

        loss_values.push_back(loss.template item<float>());

        if (batch_idx++ % 100 == 0) {
            std::cout << "Epoch: " << epoch
                      << " [" << batch_idx * batch.data.size(0) << "/60000]"
                      << " Loss: " << loss.template item<float>() << std::endl;
        }
    }
}

void plot_loss(const std::vector<float>& loss_values) {
    plt::figure_size(800, 600);
    plt::plot(loss_values);
    plt::title("Training Loss over Batches");
    plt::xlabel("Batch Number");
    plt::ylabel("Loss");
    plt::show();
}

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << " Using device: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;

    Net model;
    model.to(device);

    const int64_t batch_size = 64;
    const size_t epochs = 10;

    auto train_dataset = torch::data::datasets::MNIST("../data")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto test_dataset = torch::data::datasets::MNIST("../data", torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(test_dataset), batch_size);

    torch::optim::SGD optimizer(model.parameters(), 0.01);

    std::vector<float> loss_values;

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        train(model, *train_loader, optimizer, device, loss_values, epoch);
        evaluate(model, *test_loader, device);
    }

    // Save the model
    const std::string model_path = "./models/mnist_model.pt";
    std::filesystem::create_directories("./models");
    {
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to(model_path);
        std::cout << "Model saved to " << model_path << std::endl;
    }

    // Load and evaluate the model
    Net loaded_model;
    {
        torch::serialize::InputArchive archive;
        archive.load_from(model_path);
        loaded_model.load(archive);
        loaded_model.to(device);
        std::cout << "Model saved to " << model_path << std::endl;
    }
    evaluate(loaded_model, *test_loader, device);

    plot_loss(loss_values);

    return 0;
}
