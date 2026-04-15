//
// Created by evgen on 10.01.2025.
//

#ifndef FIELDCALCULATION_HPP
#define FIELDCALCULATION_HPP

#include "mesh/SurfaceMesh.hpp"

#include "types/Types.hpp"

#include "math/MathConstants.hpp"
#include "math/fields/SurfaceVectorField.hpp"
#include "math/integration/gauss_quadrature/GaussLegenderPoints.hpp"

#include "operators/OperatorK.hpp"
#include "operators/OperatorR.hpp"

namespace EMW::Rupor {

inline Types::Vector3c operatorK_in_point(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::point_t &point) {
    return EMW::OperatorK::K1<DefiniteIntegrals::GaussLegendre::Quadrature<4, 4>>(point, k, field) +
           EMW::OperatorK::K0<DefiniteIntegrals::GaussLegendre::Quadrature<4>>(point, k, field);
}

inline Math::SurfaceVectorField operatorK(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::SurfaceMesh &targetMesh) {

    const auto analytical = [field, k](const Types::Vector3d &point) -> Types::Vector3c {
        return operatorK_in_point(field, k, point);
    };
    return Math::SurfaceVectorField(targetMesh, analytical);
}

inline Types::Vector3c operatorR_in_point(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::point_t &point) {
    return EMW::OperatorR::detail::R<DefiniteIntegrals::GaussLegendre::Quadrature<8, 8>>(point, k, field);
}

inline Math::SurfaceVectorField operatorR(const Math::SurfaceVectorField &field, const Types::complex_d k,
                                          const Mesh::SurfaceMesh &targetMesh) {
    const auto analytical = [field, k](const Types::Vector3d &point) -> Types::Vector3c {
        return operatorR_in_point(field, k, point);
    };
    return Math::SurfaceVectorField(targetMesh, analytical);
}

inline Types::Vector3c getE_in_point(const Math::SurfaceVectorField &j_e, const Math::SurfaceVectorField &j_m,
                                     const Types::complex_d k, const Mesh::point_t &point) {
    const Types::complex_d mul = Math::Constants::i / k;
    return mul * operatorK_in_point(j_e, k, point) - operatorR_in_point(j_m, k, point);
}
inline Math::SurfaceVectorField getE(const Math::SurfaceVectorField &j_e, const Math::SurfaceVectorField &j_m,
                                     const Types::complex_d k, const Mesh::SurfaceMesh &targetMesh) {
    const Types::complex_d mul = Math::Constants::i / k;
    return mul * operatorK(j_e, k, targetMesh) - operatorR(j_m, k, targetMesh);
}

} // namespace EMW::Rupor

#endif // FIELDCALCULATION_HPP
