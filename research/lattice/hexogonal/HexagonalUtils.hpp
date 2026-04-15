//
// Created by evgen on 01.07.2025.
//

#ifndef HEXAGONALUTILS_HPP
#define HEXAGONALUTILS_HPP

#include "types/Types.hpp"

#include "geometry/PeriodicStructure.hpp"

using namespace EMW;

namespace Research::Lattice::Hexagonal {

/**
* Строит вектор из фаз, линейно расположенных относительно нуля
* для конкретной периодической структуры
*
* @param direction -- направление вектора нормали к прямой, лежащей в плоскости
*                     периодической структуры. Эта линия показывает переход из положительных фаз в отрицательные.
*                     По факту задает плоскость.
* @param phi -- насколько сильно мы хотим повернуть диаграмму направленности, в градусах
*/
template<Types::index N1, Types::index N2>
Containers::array<Types::complex_d, N1 * N2> get_linear_phase_factors(const Geometry::PeriodicStructure<N1, N2>& structure, const Types::Vector3d& direction, Types::scalar phi) {
    using persrt = Geometry::PeriodicStructure<N1, N2>;
    Containers::array<Types::complex_d, N1 * N2> phase_factors;
    for (Types::index i = 0; i < N1; i++)
        for (Types::index j = 0; j < N2; j++) {
            const Types::scalar phase_value = phi * direction.dot(structure.get_origin_matrix()[i][j].normalized());  // угол в градусах
            const Types::index index_in_phase_array = persrt::double_to_linear(i, j);
            phase_factors[index_in_phase_array] = std::exp(Math::Constants::i * phase_value * Math::Constants::deg_to_rad<Types::scalar>());
        }
    return phase_factors;
}

}

#endif //HEXAGONALUTILS_HPP
