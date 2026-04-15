//
// Created by evgen on 06.08.2025.
//

#ifndef OPERATORT_SIMPLE_HPP
#define OPERATORT_SIMPLE_HPP

#include "math/Productions.hpp"
#include "math/integration/Quadrature.hpp"
#include "math/integration/analytical/SingularIntegration.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "operators/Functions.hpp"

#include "types/Types.hpp"

#include "mesh/MeshTypes.hpp"
#include "mesh/SurfaceMesh.hpp"

#include <cmath>
#include <iostream>

/**
 * Дискретизация скалярного оператора T методом кусочно-постоянных аппроксимаций на некоторой сетке.
 * В данном подоходе оператор T заменяется матрицей, действующей на вектор,
 * соответствующий кусочно-постоянным значениям функций на ячейках сетки.
 *
 * Таким образом схема: функция кусочно-постоянная, метод коллокаций
 */
namespace EMW::Operators {


class T_operator_on_plane {
    using mesh_t = Mesh::SurfaceMesh;
    using cell_t = Mesh::IndexedCell;
    using point_t = Mesh::point_t;
    using vector_t = Types::VectorXc;
    using matrix_t = Types::MatrixXc;

    Types::complex_d k_;
    Types::scalar h_;

    /**
     *  Функция для расчета криволинейной части для оператора T на одной ячейке
     */
    template <typename Quadrature>
    Types::complex_d T_curved_part_over_cell(const point_t &point, const cell_t &cell) const {
        // Задаем функции одной переменной: векторное поле на краю ячейки
        const auto AB = [&](Types::scalar t) -> Types::Vector3c {
            const Types::Vector3d y = cell.parametrization(t, 0);
            return -Helmholtz::V(k_, point, y);
        };
        const auto BC = [&](Types::scalar t) -> Types::Vector3c {
            const Types::Vector3d y = cell.parametrization(1, t);
            return -Helmholtz::V(k_, point, y);
        };
        const auto CD = [&](Types::scalar t) -> Types::Vector3c {
            const Types::Vector3d y = cell.parametrization(1 - t, 1);
            return -Helmholtz::V(k_, point, y);
        };
        const auto DA = [&](Types::scalar t) -> Types::Vector3c {
            const Types::Vector3d y = cell.parametrization(0, 1 - t);
            return -Helmholtz::V(k_, point, y);
        };

        // Интегрируем вектор-функции по краю, а затем умножаем
        // на касательный вектор (так как он одинаков для каждого ребра)

        return Math::quasiDot(DefiniteIntegrals::integrate<Quadrature>(AB, {0}, {1}),
                              cell.integrationParameters.mul[0]) +
               Math::quasiDot(DefiniteIntegrals::integrate<Quadrature>(BC, {0}, {1}),
                              cell.integrationParameters.mul[1]) +
               Math::quasiDot(DefiniteIntegrals::integrate<Quadrature>(CD, {0}, {1}),
                              cell.integrationParameters.mul[2]) +
               Math::quasiDot(DefiniteIntegrals::integrate<Quadrature>(DA, {0}, {1}),
                              cell.integrationParameters.mul[3]);
    }

    /**
     *  Функция для расчета поверхностной части для оператора T на одной ячейке
     */
    template <typename Quadrature>
    Types::complex_d T_surface_part_over_cell(const point_t point, const cell_t &cell) const {
        if ((cell.collPoint_ - point).norm() < h_) {

            const auto phi_bounded = [&](Types::scalar p, Types::scalar q) -> Types::complex_d {
                const Types::Vector3d y = cell.parametrization(p, q);
                const Types::scalar mul = cell.multiplier(p, q);
                return Helmholtz::F_bounded_part(k_, point, y) * mul;
            };

            const auto numeric_part = DefiniteIntegrals::integrate<Quadrature>(phi_bounded, {0, 0}, {1., 1.});
            const auto analytic_part = Math::Constants::inverse_4PI<Types::scalar>() *
                                       Math::AnalyticalIntegration::integrate_1_div_r(point, cell);

            return k_ * k_ * (numeric_part + analytic_part);
        }

        const auto phi_full = [&](Types::scalar p, Types::scalar q) -> Types::complex_d {
            const Types::Vector3d y = cell.parametrization(p, q);
            const Types::scalar mul = cell.multiplier(p, q);
            return Helmholtz::F(k_, point, y) * mul;
        };

        return k_ * k_ * DefiniteIntegrals::integrate<Quadrature>(phi_full, {0, 0}, {1., 1.});
    }

  public:
    T_operator_on_plane(Types::complex_d k, Types::scalar h) : k_(k), h_(h){};

    [[nodiscard]] matrix_t matrix(const mesh_t &mesh_with_point, const mesh_t &mesh_to_integrate) const {
        using quadrature_1d = DefiniteIntegrals::GaussLegendre::Quadrature<4>;
        using quadrature_2d = DefiniteIntegrals::GaussLegendre::Quadrature<4, 4>;
        Types::index N = mesh_with_point.getCells().size();
        const Types::index M = mesh_to_integrate.getCells().size();
        Types::MatrixXc result = Types::MatrixXc::Zero(N, M);

        const auto &cells_int = mesh_to_integrate.getCells();
        const auto &cells_col = mesh_with_point.getCells();

#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                const auto fp = T_curved_part_over_cell<quadrature_1d>(cells_col[i].collPoint_, cells_int[j]);
                const auto sp = T_surface_part_over_cell<quadrature_2d>(cells_col[i].collPoint_, cells_int[j]);
                result(i, j) = 2. * (fp - sp);
                if (std::isnan(fp.real()))
                    std::cout << "fp " << i << ' ' << j << std::endl;
                if (std::isnan(sp.real()))
                    std::cout << "sp " << i << ' ' << j << std::endl;
            }
        }
        return result;
    }
};

} // namespace EMW::Operators

namespace EMW::OperatorT::LaplaceEquation {
template <typename Quadrature>
Types::scalar T_over_single_cell(const Types::Vector3d point, const Mesh::IndexedCell &cell) {
    // Задаем функции одной переменной: векторное поле на краю ячейки
    const auto AB = [&](Types::scalar t) -> Types::Vector3d {
        const Types::Vector3d y = cell.parametrization(t, 0);
        return Laplace::gradF(point, y);
    };
    const auto BC = [&](Types::scalar t) -> Types::Vector3d {
        const Types::Vector3d y = cell.parametrization(1, t);
        return Laplace::gradF(point, y);
    };
    const auto CD = [&](Types::scalar t) -> Types::Vector3d {
        const Types::Vector3d y = cell.parametrization(1 - t, 1);
        return Laplace::gradF(point, y);
    };
    const auto DA = [&](Types::scalar t) -> Types::Vector3d {
        const Types::Vector3d y = cell.parametrization(0, 1 - t);
        return Laplace::gradF(point, y);
    };
    // Интегрируем вектор-функции по краю, а зачем домножаем на касательный вектор (так как он всюду одинаков)
    return DefiniteIntegrals::integrate<Quadrature>(AB, {0}, {1}).dot(cell.integrationParameters.mul[0]) +
           DefiniteIntegrals::integrate<Quadrature>(BC, {0}, {1}).dot(cell.integrationParameters.mul[1]) +
           DefiniteIntegrals::integrate<Quadrature>(CD, {0}, {1}).dot(cell.integrationParameters.mul[2]) +
           DefiniteIntegrals::integrate<Quadrature>(DA, {0}, {1}).dot(cell.integrationParameters.mul[3]);
}

Types::MatrixXd T_over_mesh(const Mesh::SurfaceMesh &mesh_with_collocation_point,
                            const Mesh::SurfaceMesh &mesh_to_integrate) {
    const Types::index N = mesh_with_collocation_point.getCells().size();
    const Types::index M = mesh_to_integrate.getCells().size();
    Types::MatrixXd result = Types::MatrixXd::Zero(N, M);

    const auto &cells_int = mesh_to_integrate.getCells();
    const auto &cells_col = mesh_with_collocation_point.getCells();

#pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            result(i, j) = T_over_single_cell<EMW::DefiniteIntegrals::GaussLegendre::Quadrature<5>>(
                cells_col[i].collPoint_, cells_int[j]);
        }
    }
    return result;
}
} // namespace EMW::OperatorT::LaplaceEquation

#endif //OPERATORT_SIMPLE_HPP
