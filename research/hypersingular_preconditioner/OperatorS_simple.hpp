//
// Created by evgen on 04.09.2025.
//

#ifndef OPERATORS_SIMPLE_HPP
#define OPERATORS_SIMPLE_HPP

#include "math/Productions.hpp"
#include "math/integration/Quadrature.hpp"
#include "math/integration/analytical/SingularIntegration.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "operators/Functions.hpp"

#include "types/Types.hpp"

#include "mesh/MeshTypes.hpp"
#include "mesh/SurfaceMesh.hpp"

/**
 * Дискретизация скалярного оператора T методом кусочно-постоянных аппроксимаций на некоторой сетке.
 * В данном подоходе оператор T заменяется матрицей, действующей на вектор из констант,
 * соответствующим кусочно-постоянным значениям функций на ячейках сетки.
 *
 * Таким образом схема: функция кусочно-постоянная, оператор гиперсингулярный
 */
namespace EMW::OperatorS::LaplaceEquation {
/**
 * Расчет поверхностного интеграла оператора S по ячейке
 */
inline Types::scalar S_over_single_cell(const Types::Vector3d point, const Mesh::IndexedCell &cell) {
    return Math::Constants::inverse_4PI<Types::scalar>() * Math::AnalyticalIntegration::integrate_1_div_r(point, cell);
}

Types::MatrixXd S_over_mesh(const Mesh::SurfaceMesh &mesh_with_collocation_point,
                            const Mesh::SurfaceMesh &mesh_to_integrate) {
    const Types::index N = mesh_with_collocation_point.getCells().size();
    const Types::index M = mesh_to_integrate.getCells().size();
    Types::MatrixXd result = Types::MatrixXd::Zero(N, M);

    const auto &cells_int = mesh_to_integrate.getCells();
    const auto &cells_col = mesh_with_collocation_point.getCells();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result(i, j) = S_over_single_cell(cells_col[i].collPoint_, cells_int[j]);
        }
    }
    return result;
}
} // namespace EMW::OperatorS::LaplaceEquation

namespace EMW::OperatorS::Helmholtz {
/**
 * Расчет поверхностного интеграла оператора S по ячейке
 */
template<typename Quadrature>
Types::complex_d S_over_single_cell(const Types::complex_d k, const Types::Vector3d point, const Mesh::IndexedCell &cell) {
    const auto phi = [&](Types::scalar p, Types::scalar q) -> Types::complex_d {
        const Types::Vector3d y = cell.parametrization(p, q);
        const Types::scalar mul = cell.multiplier(p, q);
        return EMW::Helmholtz::F_bounded_part(k, point, y) * mul;
    };
    return DefiniteIntegrals::integrate<Quadrature>(phi, {0, 0}, {1., 1.}) +
            Math::Constants::inverse_4PI<Types::scalar>() *
                Math::AnalyticalIntegration::integrate_1_div_r(point, cell);
}

Types::MatrixXc S_over_mesh(const Types::complex_d k, const Mesh::SurfaceMesh &mesh_with_collocation_point,
                            const Mesh::SurfaceMesh &mesh_to_integrate) {
    const Types::index N = mesh_with_collocation_point.getCells().size();
    const Types::index M = mesh_to_integrate.getCells().size();
    Types::MatrixXc result = Types::MatrixXd::Zero(N, M);

    const auto &cells_int = mesh_to_integrate.getCells();
    const auto &cells_col = mesh_with_collocation_point.getCells();

    using quadrature_2d = DefiniteIntegrals::GaussLegendre::Quadrature<4, 4>;

//#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result(i, j) = S_over_single_cell<quadrature_2d>(k, cells_col[i].collPoint_, cells_int[j]);
        }
    }
    return result;
}
} // namespace EMW::OperatorS::LaplaceEquation

#endif //OPERATORS_SIMPLE_HPP
