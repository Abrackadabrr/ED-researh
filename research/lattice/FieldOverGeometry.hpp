//
// Created by evgen on 06.02.2025.
//

#ifndef FIELDOVERGEOMETRY_HPP
#define FIELDOVERGEOMETRY_HPP

#include "types/Types.hpp"
#include "consepts/Consepts.hpp"

namespace Research::Lattice {
using namespace EMW;

template <Concepts::GeomtricalStructure geometry_t> class FieldOver {
    using mesh_t = typename geometry_t::mesh_t;
    using field_t = Math::SurfaceVectorField;

    Containers::vector<field_t> electric_fields;
    Containers::vector<field_t> magnetic_fields;

    const geometry_t &geometry_ref;
    Containers::vector<mesh_t> additional_meshes;

  public:
    FieldOver(const geometry_t &geometry, Types::VectorXc &&slae_solution) : geometry_ref(geometry) {
        // подготовка к заполнению данных для полей
        electric_fields.reserve(geometry_ref.size());

        additional_meshes.reserve(geometry_ref.size());
        magnetic_fields.reserve(geometry_ref.size());
        const Types::VectorXc &j_vec{std::move(slae_solution)};

        const mesh_t &mesh_instance = geometry_ref.get(0);
        const Types::index electric_size = 2 * mesh_instance.getCells().size();
        const Types::index magnetic_size =
            2 * mesh_instance.getSubmeshInfo(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION).cells_size;
        const Types::index block_size = electric_size + magnetic_size;

        // прописываем электрические и магнитные токи сразу вместе
        for (Types::index i = 0; i < geometry_ref.size(); ++i) {
            // дампим поле электрических токов
            electric_fields.emplace_back(field_t::TangentField(
                geometry_ref.get(i), j_vec.block(i * block_size, 0, electric_size, 1), "electric_current"));
            // делаем подмногообразие для поля магнитных токов
            additional_meshes.push_back(
                geometry_ref.get(i).getSubmesh(Mesh::IndexedCell::Tag::WAVEGUIDE_CROSS_SECTION));
            // дампим в новое подмногообразие магнитный ток
            magnetic_fields.emplace_back(field_t::TangentField(
                additional_meshes[i], j_vec.block(i * block_size + electric_size, 0, magnetic_size, 1),
                "magnetic_current"));
        }
    };

    [[nodiscard]] const Containers::vector<field_t> &get_electric_fields() const { return electric_fields; }
    [[nodiscard]] const Containers::vector<field_t> &get_magnetic_fields() const { return magnetic_fields; }
    [[nodiscard]] Types::index size() const {
        if (electric_fields.size() == magnetic_fields.size()) return electric_fields.size();
        throw std::runtime_error("Error in FieldsOver fucntion size");
    }
};
}

#endif // FIELDOVERGEOMETRY_HPP
