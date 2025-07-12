#include <torch/torch.h>
#include <torch/script.h> // for torch::jit::trace
#include <memory>          // for std::make_shared
#include <iostream>
#include <filesystem>

#include "/home/ashishverma/Documents/cppmnist/src/include/evaluate.hpp"
#include "/home/ashishverma/Documents/cppmnist/src/include/net.hpp"

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

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        size_t batch_idx = 0;
        model.train();

        for (auto& batch : *train_loader) {
            optimizer.zero_grad();
            auto data = batch.data.to(device);
            auto targets = batch.target.to(device);

            auto output = model.forward(data);
            auto loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();

            if (batch_idx++ % 100 == 0) {
                std::cout << "Epoch: " << epoch
                          << " [" << batch_idx * batch.data.size(0) << "/60000]"
                          << " Loss: " << loss.item<float>() << std::endl;
            }
        }

        evaluate(model, *test_loader, device);
    }

    // Ensure model directory exists
    std::filesystem::create_directories("./models");

    // === Save the model ===
    {
        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to("./models/mnist_model.pt");
        std::cout << "Model saved to ./models/mnist_model.pt\n";
    }

    // === Load the model ===
    Net loaded_model;
    {
        torch::serialize::InputArchive archive;
        archive.load_from("./models/mnist_model.pt");
        loaded_model.load(archive);
        loaded_model.to(device);
        std::cout << "Loaded model, evaluating:\n";
    }

    evaluate(loaded_model, *test_loader, device);

    return 0;
}
