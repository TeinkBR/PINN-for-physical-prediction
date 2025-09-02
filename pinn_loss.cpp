#include <torch/extension.h>
#include <thread>
#include <vector>
#include <iostream>
#include <mutex>
#include <cmath>
#include <pybind11/stl.h>
// pybind11 headers are included by torch/extension.h; keep stl helper for std containers
namespace py = pybind11;

// Define ResPINN struct
struct ResPINN : torch::nn::Module {
    ResPINN(int num_hidden_layers = 12, int num_neurons = 128, torch::Device device = torch::kCPU) {
        input_layer = register_module("input_layer", torch::nn::Linear(6, num_neurons));
        for (int i = 0; i < num_hidden_layers; ++i) {
            auto layer = torch::nn::Sequential(
                torch::nn::Linear(num_neurons, num_neurons),
                torch::nn::Tanh()
            );
            hidden_layers.push_back(register_module("hidden_" + std::to_string(i), layer));
        }
        output_layer = register_module("output_layer", torch::nn::Linear(num_neurons, 1));
        to(device); // Move model to specified device
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(input_layer->forward(x));
        for (auto& layer : hidden_layers) {
            auto residual = x;
            x = layer->forward(x);
            x = x + residual;
        }
        return output_layer->forward(x);
    }

    torch::nn::Linear input_layer{nullptr};
    std::vector<torch::nn::Sequential> hidden_layers;
    torch::nn::Linear output_layer{nullptr};
};

// Physics functions (placeholders - update to match physics.py)
torch::Tensor direction_vector(const torch::Tensor& phi, const torch::Tensor& theta) {
    auto s_x = torch::sin(theta) * torch::cos(phi);
    auto s_y = torch::sin(theta) * torch::sin(phi);
    auto s_z = torch::cos(theta);
    return torch::cat({s_x, s_y, s_z}, 1);
}

torch::Tensor kappa_e(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z) {
    return torch::ones_like(x) * 0.1; // Placeholder
}

torch::Tensor I_b(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z, const torch::Tensor& lambda_) {
    return torch::ones_like(x); // Placeholder
}

torch::Tensor epsilon_lambda_w(const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& z,
                               const torch::Tensor& phi, const torch::Tensor& theta, const torch::Tensor& lambda_) {
    return torch::ones_like(x) * 0.5; // Placeholder
}

// Function to load model state dictionary
void load_model_state(ResPINN& model, const std::string& state_dict_path, torch::Device device) {
    torch::OrderedDict<std::string, torch::Tensor> state_dict;
    torch::load(state_dict, state_dict_path, device);
    model.load_state_dict(state_dict);
}

// Mutex for thread-safe updates
std::mutex loss_mutex;

// Compute loss for a chunk
void compute_loss_chunk(ResPINN& model,
                        const torch::Tensor& x_col_chunk,
                        const torch::Tensor& x_bc_chunk,
                        const std::vector<float>& weights,
                        float ri_mean, float ri_std,
                        torch::Tensor& loss_out,
                        std::vector<torch::Tensor>& losses_out,
                        torch::Device device) {
    const float KAPPA_SCALE = 1.0; // Adjust as needed

    // Prepare collocation points
    int batch_size = x_col_chunk.size(0);
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat);
    const float pi = std::acos(-1.0);
    auto phi = torch::rand({1}, options) * 2 * pi;
    auto theta = torch::rand({1}, options) * pi;
    auto lambda_ = torch::full({batch_size, 1}, 1e-6, options);
    phi = phi.repeat({batch_size, 1});
    theta = theta.repeat({batch_size, 1});
    auto x_col_full = torch::cat({x_col_chunk, phi, theta, lambda_}, 1).requires_grad_(true);

    // Prepare boundary points
    int batch_size_bc = x_bc_chunk.size(0);
    auto phi_bc = phi.narrow(0, 0, batch_size_bc);
    auto theta_bc = theta.narrow(0, 0, batch_size_bc);
    auto lambda_bc = torch::full({batch_size_bc, 1}, 1e-6, options);
    auto x_bc_full = torch::cat({x_bc_chunk, phi_bc, theta_bc, lambda_bc}, 1).requires_grad_(true);

    // Forward pass
    auto I_col = model.forward(x_col_full);
    auto I_bc = model.forward(x_bc_full);

    // Extract coordinates and compute gradients
    auto x = x_col_full.slice(1, 0, 1);
    auto y = x_col_full.slice(1, 1, 2);
    auto z = x_col_full.slice(1, 2, 3);
    phi = x_col_full.slice(1, 3, 4);
    theta = x_col_full.slice(1, 4, 5);
    lambda_ = x_col_full.slice(1, 5, 6);

    auto grad_I = torch::autograd::grad({I_col}, {x_col_full}, {torch::ones_like(I_col)}, true)[0];
    auto dI_dx = grad_I.slice(1, 0, 1);
    auto dI_dy = grad_I.slice(1, 1, 2);
    auto dI_dz = grad_I.slice(1, 2, 3);

    // Compute transport equation loss term (L_te)
    auto s = direction_vector(phi, theta);
    auto s_dot_grad_I = s.slice(1, 0, 1) * dI_dx + s.slice(1, 1, 2) * dI_dy + s.slice(1, 2, 3) * dI_dz;
    auto kappa_e_val = kappa_e(x, y, z) / KAPPA_SCALE;
    auto I_b_val = I_b(x, y, z, lambda_);
    auto I_b_norm = (I_b_val - ri_mean) / ri_std;
    auto I_col_norm = I_col;
    auto L_te = s_dot_grad_I + kappa_e_val * (I_col_norm - I_b_norm);

    // Compute boundary condition loss term (L_bc)
    auto x_bc = x_bc_full.slice(1, 0, 1);
    auto y_bc = x_bc_full.slice(1, 1, 2);
    auto z_bc = x_bc_full.slice(1, 2, 3);
    auto epsilon_bc = epsilon_lambda_w(x_bc, y_bc, z_bc, phi_bc, theta_bc, lambda_bc);
    auto I_b_w_bc = torch::zeros_like(I_bc);
    auto L_bc = I_bc - epsilon_bc * I_b_w_bc;

    // Compute penalty term (L_pen)
    torch::Tensor L_pen = torch::tensor(0.0, options);
    int param_count = 0;
    for (auto& param : model.named_parameters()) {
        if (param.name().find("weight") != std::string::npos) {
            L_pen = L_pen + torch::sum(param.value().pow(2));
            param_count += param.value().numel();
        }
    }
    if (param_count > 0) {
        L_pen = L_pen / param_count;
    }
    L_pen = torch::clamp(L_pen, c10::nullopt, 1e6);

    // Aggregate losses
    float w_te = weights[0], w_bc = weights[1], w_pen = weights[2];
    auto loss_te = torch::mean(L_te.pow(2));
    auto loss_bc = torch::mean(L_bc.pow(2));
    auto loss = w_te * loss_te + w_bc * loss_bc + w_pen * L_pen;

    // Store results thread-safely
    {
        std::lock_guard<std::mutex> lock(loss_mutex);
        loss_out = loss_out + loss;
        losses_out[0] = losses_out[0] + loss_te;
        losses_out[1] = losses_out[1] + loss_bc;
        losses_out[2] = losses_out[2] + L_pen;
    }
}

// Updated compute_loss to return a tuple
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_loss(
    torch::Tensor x_col, torch::Tensor x_bc,
    const std::vector<float>& weights, float ri_mean, float ri_std,
    const std::string& model_state_path,
    int num_threads = 4, std::string device_type = "cpu") {
    torch::Device device = torch::kCPU;
    if (device_type == "mps" && torch::hasMPS()) {
        device = torch::kMPS;
        num_threads = 1; // Single-threaded for MPS
    } else if (device_type == "mps") {
        std::cerr << "MPS not available, falling back to CPU" << std::endl;
    }

    ResPINN model(12, 128, device);
    load_model_state(model, model_state_path, device);

    x_col = x_col.to(device);
    x_bc = x_bc.to(device);

    int batch_size = x_col.size(0);
    int chunk_size = batch_size / num_threads;
    auto options = torch::TensorOptions().device(device).dtype(torch::kFloat);
    torch::Tensor total_loss = torch::tensor(0.0, options);
    std::vector<torch::Tensor> total_losses = {
        torch::tensor(0.0, options),
        torch::tensor(0.0, options),
        torch::tensor(0.0, options)
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? batch_size : start + chunk_size;
        auto x_col_chunk = x_col.slice(0, start, end);
        auto x_bc_chunk = x_bc.slice(0, start, std::min(end, int(x_bc.size(0))));
        threads.emplace_back(compute_loss_chunk, std::ref(model), x_col_chunk, x_bc_chunk,
                             weights, ri_mean, ri_std, std::ref(total_loss),
                             std::ref(total_losses), device);
    }

    for (auto& t : threads) {
        t.join();
    }

    total_loss = total_loss / num_threads;
    total_losses[0] = total_losses[0] / num_threads;
    total_losses[1] = total_losses[1] / num_threads;
    total_losses[2] = total_losses[2] / num_threads;

    return {total_loss, total_losses[0], total_losses[1], total_losses[2]};
}

// Python binding
PYBIND11_MODULE(pinn_loss, m) {
    m.def("compute_loss", [](torch::Tensor x_col, torch::Tensor x_bc,
                             std::vector<float> weights, float ri_mean, float ri_std,
                             std::string model_state_path, int num_threads, std::string device_type) {
        return compute_loss(x_col, x_bc, weights, ri_mean, ri_std, model_state_path, num_threads, device_type);
    }, "Compute PINN loss with multi-threading",
       py::arg("x_col"), py::arg("x_bc"), py::arg("weights"), py::arg("ri_mean"),
       py::arg("ri_std"), py::arg("model_state_path"), py::arg("num_threads") = 4,
       py::arg("device_type") = "cpu");
}