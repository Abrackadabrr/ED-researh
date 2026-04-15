//
// Created by evgen on 30.01.24.
//

#ifndef ELECTROMAGNETIC_WAVES_SCATTERING_PLATEGRID_HPP
#define ELECTROMAGNETIC_WAVES_SCATTERING_PLATEGRID_HPP

#include <utility>
#include <tuple>
#include <ranges>

#include "mesh/SurfaceMesh.hpp"
#include "mesh/Utils.hpp"
#include "types/Types.hpp"

#include <iostream>

namespace EMW::Examples::Plate {
    inline EMW::Mesh::SurfaceMesh generatePlatePrimaryMesh(int N, Types::scalar h) {
        std::vector<Mesh::point_t> meshgrid;
        meshgrid.reserve(N * N);
        Mesh::Utils::cartesian_productYZ(std::ranges::views::iota(0, N), std::ranges::views::iota(0, N),
                          std::back_inserter(meshgrid), N, h);

        const auto cellsView = std::views::iota(0, (N - 1) * (N - 1)) | std::views::transform(
                [N](int index) {
                    Types::index i = index + index / (N - 1);
                    const auto point = Mesh::IndexedCell::nodes_t{i, i + 1, i + 1 + N, i + N};
                    return point;
                }
        );

        const auto cells = Containers::vector<Mesh::IndexedCell::nodes_t>{std::ranges::begin(cellsView),
                                                                          std::ranges::end(cellsView)};

        return Mesh::SurfaceMesh{meshgrid, cells};
    }

    inline EMW::Mesh::SurfaceMesh
    generateRectangularMesh(int N1, int N2, Types::scalar h1, Types::scalar h2) {
        std::vector<Mesh::point_t> meshgrid;
        meshgrid.reserve(N1 * N2);
        Mesh::Utils::cartesian_product_unevenYZ(std::ranges::views::iota(0, N1), std::ranges::views::iota(0, N2),
                                 std::back_inserter(meshgrid), N1, N2, h1, h2);

        const auto cellsView = std::views::iota(0, (N1 - 1) * (N2 - 1)) | std::views::transform(
                [N1](int index) {
                    Types::index i = index + index / (N1 - 1);  // левый нижний индекс зависит от номера СТРОКИ
                    // в которой располагается ячейка, поэтому
                    // здесь стоит N1
                    const auto point = Mesh::IndexedCell::nodes_t{i, i + 1, i + 1 + N1, i + N1};
                    return point;
                }
        );

        const auto cells = Containers::vector<Mesh::IndexedCell::nodes_t>{std::ranges::begin(cellsView),
                                                                          std::ranges::end(cellsView)};
        return Mesh::SurfaceMesh{meshgrid, cells};
    }
}

#endif //ELECTROMAGNETIC_WAVES_SCATTERING_PLATEGRID_HPP
