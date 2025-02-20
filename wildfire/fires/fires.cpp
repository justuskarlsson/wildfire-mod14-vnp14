#include <torch/script.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/Parallel.h>

#include <queue>
#include <tuple>
#include <vector>
#include <optional>
#include <iostream>

namespace py = pybind11;
using namespace py::literals;

int UNKNOWN = 0;
int NON_FIRE = 1;
int FIRE = 2;

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b)
{
    return a + b;
}

torch::Tensor get_deltas(int t_range, int y_range, int x_range)
{
    int n = (2 * t_range + 1) * (2 * x_range + 1) * (2 * y_range + 1) - 1;
    torch::Tensor deltas = torch::zeros({n, 3}, torch::dtype(torch::kInt32));
    int i = 0;
    for (int t = -t_range; t < t_range + 1; t++)
    {
        for (int y = -y_range; y < y_range + 1; y++)
        {
            for (int x = -x_range; x < x_range + 1; x++)
            {
                if (t == 0 && x == 0 && y == 0)
                    continue;
                deltas[i][0] = t;
                deltas[i][1] = y;
                deltas[i][2] = x;
                i++;
            }
        }
    }
    return deltas;
}

struct Component
{
    int idx;
    int size;
    int num_fire;
    // bbox3d
    int t_min;
    int t_max;
    int y_min;
    int y_max;
    int x_min;
    int x_max;

    py::dict to_dict() const
    {
        return py::dict(
            "idx"_a = idx,
            "size"_a = size,
            "num_fire"_a = num_fire,
            "t_min"_a = t_min,
            "t_max"_a = t_max,
            "y_min"_a = y_min,
            "y_max"_a = y_max,
            "x_min"_a = x_min,
            "x_max"_a = x_max);
    }
};

std::optional<Component>
single_bfs(int t, int y, int x, torch::Tensor fire_cls, torch::Tensor &components, int &component_idx)
{
    // Create accessors for direct integer access
    auto fire_cls_a = fire_cls.accessor<uint8_t, 3>();
    auto components_a = components.accessor<int, 3>();

    // Dimensions for boundary checks
    int T = fire_cls.size(0);
    int H = fire_cls.size(1);
    int W = fire_cls.size(2);

    // Initialize component
    Component component{
        component_idx, // idx
        1,             // size
        1,             // num_fire
        t, t, y, y, x, x};

    // Mark visited
    components_a[t][y][x] = component_idx;

    // Precompute neighbor deltas
    torch::Tensor deltas = get_deltas(2, 1, 1);
    auto deltas_a = deltas.accessor<int, 2>();
    int n_deltas = deltas.size(0);

    // BFS
    std::queue<std::tuple<int, int, int>> q;
    q.push({t, y, x});
    while (!q.empty())
    {
        auto [ct, cy, cx] = q.front();
        q.pop();

        for (int i = 0; i < n_deltas; i++)
        {
            int t2 = ct + deltas_a[i][0];
            int y2 = cy + deltas_a[i][1];
            int x2 = cx + deltas_a[i][2];

            // Boundary check
            if (t2 < 0 || t2 >= T || y2 < 0 || y2 >= H || x2 < 0 || x2 >= W)
                continue;

            int neighbor_cls = fire_cls_a[t2][y2][x2];
            if (neighbor_cls != FIRE)
                continue;

            if (components_a[t2][y2][x2] != 0)
                continue;

            // Mark visited
            components_a[t2][y2][x2] = component_idx;
            component.size++;
            if (neighbor_cls == FIRE)
                component.num_fire++;

            // Update bounding box
            component.t_min = std::min(component.t_min, t2);
            component.t_max = std::max(component.t_max, t2);
            component.y_min = std::min(component.y_min, y2);
            component.y_max = std::max(component.y_max, y2);
            component.x_min = std::min(component.x_min, x2);
            component.x_max = std::max(component.x_max, x2);

            q.push({t2, y2, x2});
        }
    }

    component_idx++;
    return component;
}

/**
 * fire_cls: T Y X : 0-9
 * nodes: TYX 3
 */
std::tuple<torch::Tensor, std::vector<py::dict>> search(torch::Tensor fire_cls, torch::Tensor nodes)
{
    torch::Tensor components = torch::zeros(fire_cls.sizes(), torch::dtype(torch::kInt32));
    std::vector<Component> components_list;
    int component_idx = 1;
    auto nodes_a = nodes.accessor<int64_t, 2>();
    auto fire_cls_a = fire_cls.accessor<uint8_t, 3>();
    auto components_a = components.accessor<int, 3>();

    size_t num_nodes = nodes.size(0);
    for (size_t i = 0; i < num_nodes; i++)
    {
        int t = nodes_a[i][0];
        int y = nodes_a[i][1];
        int x = nodes_a[i][2];

        int fire_cls_idx = fire_cls_a[t][y][x];
        if (fire_cls_idx != FIRE || components_a[t][y][x] != 0)
            continue;

        auto component = single_bfs(t, y, x, fire_cls, components, component_idx);
        if (component.has_value())
        {
            components_list.push_back(component.value());
            // std::cout << i << "/" << num_nodes
            //           << " num_fire: " << component.value().num_fire
            //           << std::endl;
        }
    }

    std::vector<py::dict> component_dicts;
    component_dicts.reserve(components_list.size());
    for (auto &component : components_list)
    {
        component_dicts.push_back(component.to_dict());
    }

    return std::make_tuple(components, component_dicts);
}

torch::Tensor get_deltas_yx(int y_range, int x_range)
{
    int n = (2 * y_range + 1) * (2 * x_range + 1);
    torch::Tensor deltas = torch::zeros({n, 2}, torch::dtype(torch::kInt32));
    int i = 0;
    for (int y = -y_range; y < y_range + 1; y++)
    {
        for (int x = -x_range; x < x_range + 1; x++)
        {
            deltas[i][0] = y;
            deltas[i][1] = x;
            i++;
        }
    }
    return deltas;
}

template <int dim>
using tensor_at = at::TensorAccessor<float, dim>;

torch::Tensor
project(
    torch::Tensor data,
    int W,
    int H,
    torch::Tensor xy_coords,
    const std::string &method,
    int kernel_size,
    float no_data_val)
{
    bool method_inv_dist = method == "inv_dist";
    bool method_nearest = method == "nearest";
    if (!method_inv_dist && !method_nearest)
        throw std::runtime_error("Invalid method: " + method);
    data = data.permute({1, 2, 0}).contiguous();
    auto xy_coords_a = xy_coords.accessor<float, 3>(); // Y X 2
    auto data_a = data.accessor<float, 3>();           // Y X C
    int src_H = data.size(0);
    int src_W = data.size(1);
    int C = data.size(2);
    // std::cout << "H: " << H << " W: " << W << " C: " << C << std::endl;
    constexpr int block_size = 384;
    assert(xy_coords.size(0) == src_H);
    assert(xy_coords.size(1) == src_W);
    int NY = std::ceil(H / (float)block_size);
    int NX = std::ceil(W / (float)block_size);
    int num_blocks = NY * NX;
    // std::cout << "NY: " << NY << " NX: " << NX << " num_blocks: " << num_blocks << std::endl;
    int num_threads = at::get_num_threads();
    if (num_threads == 1)
    {
        int num_cores = std::thread::hardware_concurrency();
        num_cores = std::min(num_cores, 8);
        at::set_num_threads(num_cores); // Set to 8 threads
        std::cout << "Set LibTorch threads to " << num_cores << std::endl;
    }

    int element_size = 2 + C;
    int num_block_elements = 0;
    std::vector<int> block_sizes(num_blocks, 0);
    std::vector<int> block_counts(num_blocks, 0);
    std::vector<int> block_offsets(num_blocks, 0);
    std::vector<float> blocks_buffer;

    for (int t = 0; t < 2; t++)
    {
        for (int y = 0; y < src_H; y++)
        {
            for (int x = 0; x < src_W; x++)
            {
                float dst_x = xy_coords_a[y][x][0];
                float dst_y = xy_coords_a[y][x][1];

                int bx = std::floor(dst_x / block_size);
                int by = std::floor(dst_y / block_size);
                if (bx < 0 || bx >= NX || by < 0 || by >= NY)
                    continue;
                int block_idx = by * NX + bx;
                if (t == 0)
                {
                    block_sizes[block_idx] += 1;
                }
                else
                {
                    int i = (block_offsets[block_idx] + block_counts[block_idx]) * element_size;
                    blocks_buffer[i++] = dst_x;
                    blocks_buffer[i++] = dst_y;
                    for (int c = 0; c < C; c++)
                    {
                        blocks_buffer[i + c] = data_a[y][x][c];
                    }
                    block_counts[block_idx]++;
                }
            }
        }
        if (t == 0)
        {
            for (int b = 1; b < num_blocks; b++)
            {
                block_offsets[b] = block_offsets[b - 1] + block_sizes[b - 1];
            }
            num_block_elements = block_offsets[num_blocks - 1] + block_sizes[num_blocks - 1];
            blocks_buffer.resize(num_block_elements * element_size);
        }
    }
    torch::Tensor deltas = get_deltas_yx(kernel_size / 2, kernel_size / 2);
    int num_deltas = deltas.size(0);
    auto deltas_a = deltas.accessor<int, 2>();
    // std::cout << "Starting block processing" << std::endl;
    torch::Tensor out = torch::zeros({H, W, C}, torch::dtype(torch::kFloat32));
    auto out_a = out.accessor<float, 3>();
    torch::Tensor block_values = torch::full({num_blocks, block_size, block_size, C}, 0.0);
    torch::Tensor block_weights = torch::full({num_blocks, block_size, block_size, C}, 0.0);
    auto block_values_a = block_values.accessor<float, 4>();
    auto block_weights_a = block_weights.accessor<float, 4>();
    at::parallel_for(0, num_blocks, 8, [&](int64_t start, int64_t end)
                     {
        for (int i = start; i < end; i++)
        {
            int yo = (i / NX) * block_size;
            int xo = (i % NX) * block_size;
            for (int j = 0; j < block_sizes[i]; j++)
            {
                int bo = (block_offsets[i] + j) * element_size;
                float *item_ptr = &blocks_buffer[bo];
                float x = item_ptr[0] - xo;
                float y = item_ptr[1] - yo;
                for (int k = 0; k < num_deltas; k++)
                {
                    int y2 = std::round(y + deltas_a[k][0]);
                    int x2 = std::round(x + deltas_a[k][1]);
                    if (y2 < 0 || y2 >= block_size || x2 < 0 || x2 >= block_size)
                        continue;
                    float dist = std::hypotf(y2 - y, x2 - x);
                    float weight;
                    if (method_inv_dist || method_nearest)
                    {
                        weight = dist == 0 ? 1e5 : 1.0 / dist;
                    }

                    for (int c = 0; c < C; c++)
                    {
                        float val = item_ptr[2 + c];
                        if (val == no_data_val)
                            continue;
                        if (method_inv_dist)
                        {
                            block_values_a[i][y2][x2][c] += weight * val;
                            block_weights_a[i][y2][x2][c] += weight;
                        }
                        else // nearest
                        {
                            if (weight > block_weights_a[i][y2][x2][c])
                            {
                                block_values_a[i][y2][x2][c] = val;
                                block_weights_a[i][y2][x2][c] = weight;
                            }
                        }
                    }
                }
            }
            for (int dy = 0; dy < block_size; dy++)
            {
                int y = yo + dy;
                if (y >= H || y < 0)
                    break;
                for (int dx = 0; dx < block_size; dx++)
                {
                    int x = xo + dx;
                    if (x >= W || x < 0)
                        break;
                    for (int c = 0; c < C; c++)
                    {
                        if (block_weights_a[i][dy][dx][c] == 0)
                        {
                            out_a[y][x][c] = no_data_val;
                            continue;
                        }
                        float& val = out_a[y][x][c];
                        if (method_inv_dist)
                        {
                            val += block_values_a[i][dy][dx][c] / block_weights_a[i][dy][dx][c];
                        }
                        else // nearest
                        {
                            val = block_values_a[i][dy][dx][c];
                        }
                    }
                }
            }
        } });
    // std::cout << "Finished block processing" << std::endl;
    out = out.permute({2, 0, 1}).contiguous();
    return out;
}

#if defined(__INTELLISENSE__)
// Skip extension code so IntelliSense won't barf
#else
#include <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_tensors", add_tensors);
    m.def("search", search);
    m.def("project", project);
}
#endif
