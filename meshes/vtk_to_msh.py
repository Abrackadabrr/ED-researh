#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

from vtkmodules.vtkIOLegacy import vtkDataSetReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader


KEEP_SMALLEST_2D_CELLS = 20

VTK_TO_GMSH = {
    1: (15, None),  # VTK_VERTEX -> point
    3: (1, None),  # VTK_LINE -> 2-node line
    5: (2, None),  # VTK_TRIANGLE -> 3-node triangle
    7: (None, None),  # VTK_POLYGON, supported below only for 3 or 4 nodes
    8: (3, [0, 1, 3, 2]),  # VTK_PIXEL -> 4-node quadrangle
    9: (3, None),  # VTK_QUAD -> 4-node quadrangle
    10: (4, None),  # VTK_TETRA -> 4-node tetrahedron
    11: (5, [0, 1, 3, 2, 4, 5, 7, 6]),  # VTK_VOXEL -> 8-node hexahedron
    12: (5, None),  # VTK_HEXAHEDRON -> 8-node hexahedron
    13: (6, None),  # VTK_WEDGE -> 6-node prism
    14: (7, None),  # VTK_PYRAMID -> 5-node pyramid
}


def read_dataset(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".vtu":
        reader = vtkXMLUnstructuredGridReader()
    elif suffix == ".vtk":
        reader = vtkDataSetReader()
    else:
        raise ValueError(f"Unsupported input extension '{suffix}'. Expected .vtk or .vtu")

    reader.SetFileName(str(path))
    reader.Update()
    dataset = reader.GetOutput()
    if dataset is None or dataset.GetNumberOfPoints() == 0:
        raise ValueError(f"Cannot read mesh points from {path}")
    return dataset


def gmsh_cell(cell):
    vtk_type = cell.GetCellType()
    point_ids = [cell.GetPointId(i) + 1 for i in range(cell.GetNumberOfPoints())]

    if vtk_type == 7:
        if len(point_ids) == 3:
            return 2, point_ids
        if len(point_ids) == 4:
            return 3, point_ids
        raise ValueError(f"VTK_POLYGON with {len(point_ids)} nodes is not supported by this converter")

    if vtk_type not in VTK_TO_GMSH:
        raise ValueError(f"Unsupported VTK cell type {vtk_type}")

    gmsh_type, reorder = VTK_TO_GMSH[vtk_type]
    if gmsh_type is None:
        raise ValueError(f"Unsupported VTK cell type {vtk_type}")

    if reorder is not None:
        if len(point_ids) != len(reorder):
            raise ValueError(f"Unexpected node count {len(point_ids)} for VTK cell type {vtk_type}")
        point_ids = [point_ids[i] for i in reorder]

    return gmsh_type, point_ids


def dataset_to_mesh(dataset):
    nodes = [dataset.GetPoint(point_idx) for point_idx in range(dataset.GetNumberOfPoints())]
    elements = []
    for cell_idx in range(dataset.GetNumberOfCells()):
        elements.append(gmsh_cell(dataset.GetCell(cell_idx)))
    return nodes, elements


def subtract(first, second):
    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


def cross(first, second):
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def norm(vector):
    return (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]) ** 0.5


def triangle_area(nodes, first_node_id, second_node_id, third_node_id):
    first = nodes[first_node_id - 1]
    second = nodes[second_node_id - 1]
    third = nodes[third_node_id - 1]
    return 0.5 * norm(cross(subtract(second, first), subtract(third, first)))


def element_area(nodes, gmsh_type, point_ids):
    if gmsh_type == 2:
        return triangle_area(nodes, point_ids[0], point_ids[1], point_ids[2])
    if gmsh_type == 3:
        return triangle_area(nodes, point_ids[0], point_ids[1], point_ids[2]) + triangle_area(
            nodes, point_ids[0], point_ids[2], point_ids[3]
        )
    return None


def smallest_2d_element_ids(nodes, elements, count):
    cells_with_area = []
    for element_id, (gmsh_type, point_ids) in enumerate(elements, start=1):
        area = element_area(nodes, gmsh_type, point_ids)
        if area is not None:
            cells_with_area.append((area, element_id))

    cells_with_area.sort()
    return {element_id for _, element_id in cells_with_area[:count]}


def edge_midpoint(edge_to_midpoint, nodes, first_node_id, second_node_id):
    edge = tuple(sorted((first_node_id, second_node_id)))
    if edge in edge_to_midpoint:
        return edge_to_midpoint[edge]

    first = nodes[first_node_id - 1]
    second = nodes[second_node_id - 1]
    nodes.append(
        (
            0.5 * (first[0] + second[0]),
            0.5 * (first[1] + second[1]),
            0.5 * (first[2] + second[2]),
        )
    )
    midpoint_id = len(nodes)
    edge_to_midpoint[edge] = midpoint_id
    return midpoint_id


def refine_triangle(edge_to_midpoint, nodes, point_ids):
    first, second, third = point_ids
    midpoint_12 = edge_midpoint(edge_to_midpoint, nodes, first, second)
    midpoint_23 = edge_midpoint(edge_to_midpoint, nodes, second, third)
    midpoint_31 = edge_midpoint(edge_to_midpoint, nodes, third, first)

    return [
        (2, [first, midpoint_12, midpoint_31]),
        (2, [midpoint_12, second, midpoint_23]),
        (2, [midpoint_31, midpoint_23, third]),
        (2, [midpoint_12, midpoint_23, midpoint_31]),
    ]


def refine_quad(edge_to_midpoint, nodes, point_ids):
    first, second, third, fourth = point_ids
    midpoint_12 = edge_midpoint(edge_to_midpoint, nodes, first, second)
    midpoint_23 = edge_midpoint(edge_to_midpoint, nodes, second, third)
    midpoint_34 = edge_midpoint(edge_to_midpoint, nodes, third, fourth)
    midpoint_41 = edge_midpoint(edge_to_midpoint, nodes, fourth, first)

    first_point = nodes[first - 1]
    second_point = nodes[second - 1]
    third_point = nodes[third - 1]
    fourth_point = nodes[fourth - 1]
    nodes.append(
        (
            0.25 * (first_point[0] + second_point[0] + third_point[0] + fourth_point[0]),
            0.25 * (first_point[1] + second_point[1] + third_point[1] + fourth_point[1]),
            0.25 * (first_point[2] + second_point[2] + third_point[2] + fourth_point[2]),
        )
    )
    center_id = len(nodes)

    return [
        (3, [first, midpoint_12, center_id, midpoint_41]),
        (3, [midpoint_12, second, midpoint_23, center_id]),
        (3, [center_id, midpoint_23, third, midpoint_34]),
        (3, [midpoint_41, center_id, midpoint_34, fourth]),
    ]


def refine_2d_except_kept(nodes, elements, keep_elements):
    edge_to_midpoint = {}
    refined_elements = []

    for original_element_id, (gmsh_type, point_ids) in enumerate(elements, start=1):
        if original_element_id in keep_elements:
            refined_elements.append((gmsh_type, point_ids))
            continue

        if gmsh_type == 2:
            refined_elements.extend(refine_triangle(edge_to_midpoint, nodes, point_ids))
        elif gmsh_type == 3:
            refined_elements.extend(refine_quad(edge_to_midpoint, nodes, point_ids))
        else:
            refined_elements.append((gmsh_type, point_ids))

    return nodes, refined_elements


def write_msh(nodes, elements, output_path: Path):
    with output_path.open("w", encoding="utf-8") as msh:
        msh.write("$MeshFormat\n")
        msh.write("2.2 0 8\n")
        msh.write("$EndMeshFormat\n")

        msh.write("$Nodes\n")
        msh.write(f"{len(nodes)}\n")
        for point_idx, point in enumerate(nodes, start=1):
            msh.write(f"{point_idx} {point[0]:.17g} {point[1]:.17g} {point[2]:.17g}\n")
        msh.write("$EndNodes\n")

        msh.write("$Elements\n")
        msh.write(f"{len(elements)}\n")
        for element_idx, (gmsh_type, point_ids) in enumerate(elements, start=1):
            nodes = " ".join(str(point_id) for point_id in point_ids)
            msh.write(f"{element_idx} {gmsh_type} 0 {nodes}\n")
        msh.write("$EndElements\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert VTK/VTU mesh geometry to Gmsh MSH 2.2 ASCII. "
            f"All 2D cells are split except {KEEP_SMALLEST_2D_CELLS} smallest cells by area."
        )
    )
    parser.add_argument("input", type=Path, help="Input .vtk or .vtu mesh file")
    parser.add_argument("output", type=Path, nargs="?", help="Output .msh file")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else input_path.with_suffix(".msh")

    dataset = read_dataset(input_path)
    nodes, elements = dataset_to_mesh(dataset)
    keep_elements = smallest_2d_element_ids(nodes, elements, KEEP_SMALLEST_2D_CELLS)
    nodes, elements = refine_2d_except_kept(nodes, elements, keep_elements)

    write_msh(nodes, elements, output_path)
    print(
        f"Written {output_path}: {len(nodes)} nodes, {len(elements)} elements; "
        f"kept {len(keep_elements)} smallest 2D cells unsplit",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
