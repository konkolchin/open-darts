import os

import numpy as np


def write_lines_vtp(path: str, nodes_xyz: np.ndarray, output_properties: dict = None):
    """
    Write a polyline as a VTK XML PolyData file (.vtp) - convenient for well visualization.

    Nodes define N segments via consecutive pairs (node i → node i+1).
    ParaView's Tube filter requires PolyData input - this function produces it.

    :param path: Output file path (should end in .vtp).
    :param nodes_xyz: Node coordinates, shape (n_nodes, 3).
    :param output_properties: Optional dict of per-segment arrays, each length n_nodes-1.
    """
    coords = np.asarray(nodes_xyz, dtype=float)
    npts = coords.shape[0]
    nseg = npts - 1

    if output_properties is not None:
        for name, vals in output_properties.items():
            arr = np.asarray(vals).ravel()
            if arr.shape[0] != nseg:
                raise ValueError(
                    f"output_properties['{name}'] length {arr.shape[0]} != n_segments {nseg}"
                )

    connectivity = " ".join(f"{i} {i + 1}" for i in range(nseg))
    offsets = " ".join(str(2 * (i + 1)) for i in range(nseg))

    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">')
    xml.append('  <PolyData>')
    xml.append(
        f'    <Piece NumberOfPoints="{npts}" NumberOfVerts="0" '
        f'NumberOfLines="{nseg}" NumberOfStrips="0" NumberOfPolys="0">'
    )
    xml.append('      <Points>')
    xml.append(
        '        <DataArray type="Float64" NumberOfComponents="3" format="ascii">'
    )
    for pt in coords:
        xml.append(f'          {pt[0]:.10g} {pt[1]:.10g} {pt[2]:.10g}')
    xml.append('        </DataArray>')
    xml.append('      </Points>')
    xml.append('      <Lines>')
    xml.append('        <DataArray type="Int32" Name="connectivity" format="ascii">')
    xml.append(f'          {connectivity}')
    xml.append('        </DataArray>')
    xml.append('        <DataArray type="Int32" Name="offsets" format="ascii">')
    xml.append(f'          {offsets}')
    xml.append('        </DataArray>')
    xml.append('      </Lines>')
    if output_properties:
        xml.append('      <CellData>')
        for name, vals in output_properties.items():
            arr = np.asarray(vals).ravel().astype(float)
            xml.append(
                f'        <DataArray type="Float64" Name="{name}" format="ascii">'
            )
            xml.append('          ' + ' '.join(f'{v:.10g}' for v in arr))
            xml.append('        </DataArray>')
        xml.append('      </CellData>')
    xml.append('    </Piece>')
    xml.append('  </PolyData>')
    xml.append('</VTKFile>')

    with open(path, "w") as fh:
        fh.write("\n".join(xml))


def _normalize_pvd_entry(entry):
    # Reservoir solution PVDs use the compact (time, file) form because they
    # contain one dataset per timestep. Well PVDs can contain several wells at
    # the same timestep, so they pass (time, file, group, part). The group is a
    # readable label such as "well_P1"; part is the numeric dataset id ParaView
    # uses to keep those same-timestep datasets separate.
    if len(entry) == 2:
        timestep, filename = entry
        group = ""
        part = 0
    elif len(entry) == 4:
        timestep, filename, group, part = entry
    else:
        raise ValueError(
            "PVD entries must be (timestep, filename) or "
            "(timestep, filename, group, part)."
        )

    return timestep, filename, group, part


def write_pvd(path: str, entries: list[tuple]):
    """
    Write a ParaView Data (.pvd) collection file referencing a time series of VTK files.

    :param path: Output file path (should end in .pvd).
    :param entries: List of (timestep, filename) or
                    (timestep, filename, group, part) entries. Filenames are written
                    without path modification and should be relative to the directory
                    containing the .pvd file.
    """
    xml = []
    xml.append('<?xml version="1.0"?>')
    xml.append('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">')
    xml.append('  <Collection>')
    for entry in entries:
        timestep, filename, group, part = _normalize_pvd_entry(entry)
        xml.append(
            f'    <DataSet timestep="{timestep:.10g}" '
            f'group="{group}" part="{part}" file="{filename}"/>'
        )
    xml.append('  </Collection>')
    xml.append('</VTKFile>')

    with open(path, "w") as fh:
        fh.write("\n".join(xml))


def _block_shapes_match(cur_blocks, ref_blocks):
    if len(cur_blocks) != len(ref_blocks):
        return False
    return all(c.shape == r.shape for c, r in zip(cur_blocks, ref_blocks, strict=True))


def write_vtk_difference(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    vtk_diff_fname: str,
    props: list = None,
    include_values: bool = True,
    relative: bool = True,
    eps_div: float = 1e-15,
    compression: str = 'zlib',
    verbose: bool = True,
):
    """
    Write the difference between two vtk solutions into a separate vtk file.

    The output carries the mesh of the current solution and, for every compared
    property, a ``<prop>_diff`` array (current - reference). This makes a visual
    comparison of two runs a matter of opening one file in ParaView instead of
    switching back and forth between two.

    :param vtk_cur_fname: current solution.
    :param vtk_ref_fname: reference solution to subtract.
    :param vtk_diff_fname: file to write; its directory is created if missing.
    :param props: property names to compare, all properties present in both files if None.
    :param include_values: also store the compared values themselves, as ``<prop>``
                           (current) and ``<prop>_ref`` (reference).
    :param relative: also store ``<prop>_reldiff`` = diff / (|reference| + eps_div).
    :param eps_div: regularization of the relative difference denominator.
    :param compression: meshio compression of the output file, None to switch it off.
    :param verbose: print the maximum absolute and relative difference of every property.
    :return: dict prop -> (max absolute difference, max relative difference).
    """
    import meshio

    cur = meshio.read(vtk_cur_fname)
    ref = meshio.read(vtk_ref_fname)

    if cur.points.shape != ref.points.shape:
        raise ValueError(
            f'Meshes differ: {cur.points.shape[0]} points in {vtk_cur_fname} vs '
            f'{ref.points.shape[0]} in {vtk_ref_fname}; the two solutions cannot be subtracted.'
        )
    max_point_shift = (
        float(np.abs(cur.points - ref.points).max()) if cur.points.size else 0.0
    )
    if verbose and max_point_shift > 0.0:
        print(
            f'write_vtk_difference: warning: node coordinates differ, max shift {max_point_shift:.3e}'
        )

    if props is None:
        names = [name for name in cur.cell_data if name in ref.cell_data]
    else:
        names = list(props)

    stats = {}
    cell_data = {}
    for name in names:
        if name not in cur.cell_data:
            if verbose:
                print(
                    f'write_vtk_difference: no property "{name}" in {vtk_cur_fname}, skipped'
                )
            continue
        if name not in ref.cell_data:
            if verbose:
                print(
                    f'write_vtk_difference: no property "{name}" in {vtk_ref_fname}, skipped'
                )
            continue
        cur_blocks = [np.asarray(b, dtype=float) for b in cur.cell_data[name]]
        ref_blocks = [np.asarray(b, dtype=float) for b in ref.cell_data[name]]
        if not _block_shapes_match(cur_blocks, ref_blocks):
            if verbose:
                print(
                    f'write_vtk_difference: property "{name}" has a different shape in the two files, skipped'
                )
            continue

        diff_blocks = [c - r for c, r in zip(cur_blocks, ref_blocks, strict=True)]
        rel_blocks = [
            d / (np.abs(r) + eps_div)
            for d, r in zip(diff_blocks, ref_blocks, strict=True)
        ]

        if include_values:
            cell_data[name] = cur_blocks
            cell_data[name + '_ref'] = ref_blocks
        cell_data[name + '_diff'] = diff_blocks
        if relative:
            cell_data[name + '_reldiff'] = rel_blocks

        max_abs = max(
            (float(np.abs(d).max()) for d in diff_blocks if d.size), default=0.0
        )
        max_rel = max(
            (float(np.abs(d).max()) for d in rel_blocks if d.size), default=0.0
        )
        stats[name] = (max_abs, max_rel)
        if verbose:
            print(f'{name}: max abs diff {max_abs:.6e}, max rel diff {max_rel:.6e}')

    if not cell_data:
        raise ValueError(
            f'No comparable cell property found in {vtk_cur_fname} and {vtk_ref_fname}.'
        )

    diff_dir = os.path.dirname(vtk_diff_fname)
    if diff_dir:
        os.makedirs(diff_dir, exist_ok=True)
    meshio.write(
        vtk_diff_fname,
        meshio.Mesh(cur.points, cur.cells, cell_data=cell_data),
        binary=True,
        compression=compression,
    )
    if verbose:
        print('SAVED DIFFERENCE VTK FILE', vtk_diff_fname)
    return stats


# Default tolerances of the reference comparison.
REL_TOLERANCE = 1e-6
ABS_TOLERANCE = 1e-8


def read_vtk(filename: str, props: list):
    """
    Read cell centers, cell data, points and point data of the listed properties.

    :param filename: vtk file to read
    :param props: property names to read
    :return: (centers, cell_data, points, point_data)
    """
    import meshio

    mesh = meshio.read(filename=filename)

    # cell data
    centers = np.empty([0, 3])
    cell_data = {}
    for geom_name, geom in mesh.cells_dict.items():
        centers = np.append(centers, np.average(mesh.points[geom], axis=1), axis=0)
        for prop in props:
            if prop in mesh.cell_data_dict:
                if prop not in cell_data:
                    cell_data[prop] = []
                cell_data[prop].append(mesh.cell_data_dict[prop][geom_name])

    # point data
    points = mesh.points
    point_data = {}
    for prop_name, prop in mesh.point_data.items():
        if prop_name in props:
            point_data[prop_name] = prop

    return centers, cell_data, points, point_data


def save_vtk_ref(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    props: list = None,
    compression: str = 'zlib',
):
    """
    Store a computed vtk solution as the reference one.

    Keeping only the compared properties and writing the file compressed to make reference files smaller.

    :param vtk_cur_fname: computed solution to store.
    :param vtk_ref_fname: reference file to write; its directory is created if missing.
    :param props: cell properties to keep, all of them if None.
    :param compression: meshio compression of the reference file, None to switch it off.
    :return: 0
    """
    import meshio

    mesh = meshio.read(vtk_cur_fname)
    if props is None:
        cell_data = mesh.cell_data
    else:
        cell_data = {
            prop: mesh.cell_data[prop] for prop in props if prop in mesh.cell_data
        }
    ref_dir = os.path.dirname(vtk_ref_fname)
    if ref_dir:
        os.makedirs(ref_dir, exist_ok=True)
    meshio.write(
        vtk_ref_fname,
        meshio.Mesh(mesh.points, mesh.cells, cell_data=cell_data),
        binary=True,
        compression=compression,
    )
    print('SAVED REFERENCE VTK FILE', vtk_ref_fname)
    return 0


def compare_vtk_with_ref(
    vtk_cur_fname: str,
    vtk_ref_fname: str,
    props: list,
    rel_tolerance: float = REL_TOLERANCE,
    abs_tolerance: float = ABS_TOLERANCE,
    verbose: bool = True,
):
    """
    Compare a computed vtk solution with the reference one, property by property.

    Cell centers and points are compared as well, so a changed mesh is reported too.

    :param vtk_cur_fname: computed solution.
    :param vtk_ref_fname: reference solution.
    :param props: cell properties to compare; a property missing from the reference file
                  is not compared, one missing only from the solution is an error.
    :param rel_tolerance: relative tolerance of the comparison.
    :param abs_tolerance: absolute tolerance of the comparison.
    :param verbose: print the maximum relative difference of every compared item.
    :return: 0 if the solutions match, 1 otherwise.
    """
    if not os.path.exists(vtk_ref_fname):
        print('REFERENCE VTK FILE', os.path.abspath(vtk_ref_fname), 'does not exist.')
        print('Run with UPLOAD_PKL=1 to create or update this reference file.')
        return 1
    if not os.path.exists(vtk_cur_fname):
        print(
            'SOLUTION VTK FILE',
            os.path.abspath(vtk_cur_fname),
            'was not written by the run.',
        )
        return 1

    ref = read_vtk(vtk_ref_fname, props)  # the reference solution
    cur = read_vtk(vtk_cur_fname, props)  # the current solution
    names = [
        'centers',
        'cell_data',
        'points',
        'point_data',
    ]  # object names to be compared

    eps_div = 1e-15  # to avoid division by zero
    ret_flag = 0
    # the pass criterion is np.isclose: abs.diff <= abs_tolerance + rel_tolerance * |reference|
    print(
        'compare: tolerances: abs',
        abs_tolerance,
        'rel',
        rel_tolerance,
    )
    for n, r, c in zip(names, ref, cur, strict=True):
        if isinstance(r, dict):  # cell_data is a dict, so check each item there
            if len(r) == 0:  # point_data is empty, skip it
                continue
            missing = [prop for prop in r.keys() if prop not in c]
            if missing:
                print('There are no properties', missing, 'in', vtk_cur_fname)
                ret_flag = 1
            ns = [prop for prop in r.keys() if prop in c]
            rs, cs = [r[prop] for prop in ns], [c[prop] for prop in ns]  # dict to list
        else:
            ns, rs, cs = (
                [n],
                [r],
                [c],
            )  # create a list just to have a loop below for both cases
        for ni, ri, ci in zip(ns, rs, cs, strict=True):
            r1 = np.array(ri)
            c1 = np.array(ci)
            if r1.shape != c1.shape:
                print(
                    'There is a shape difference', r1.shape, 'vs', c1.shape, 'for', ni
                )
                ret_flag = 1
                continue
            abs_diff = np.fabs(r1 - c1)  # absolute difference
            rel_diff = abs_diff / (np.fabs(r1) + eps_div)  # relative difference
            ok = np.isclose(r1, c1, rtol=rel_tolerance, atol=abs_tolerance)
            # how many values exceed each tolerance on its own
            n_abs = int((abs_diff > abs_tolerance).sum())
            n_rel = int((rel_diff > rel_tolerance).sum())
            counts = (
                f'{n_abs} of {abs_diff.size} values over abs.tol, {n_rel} over rel.tol'
            )
            if ok.all():
                if verbose:
                    abs_max = abs_diff.max() if abs_diff.size else 0.0
                    rel_max = rel_diff.max() if rel_diff.size else 0.0
                    print(
                        'Comparing',
                        ni,
                        'abs.diff',
                        abs_max,
                        'rel.diff',
                        rel_max,
                        '(' + counts + ')',
                    )
            else:
                ret_flag = 1
                # report the difference over the failed values only,
                # the maximum over all values can be dominated by near-zero references
                bad = ~ok
                print(
                    'There is a difference for',
                    ni,
                    'abs.diff',
                    abs_diff[bad].max(),
                    'rel.diff',
                    rel_diff[bad].max(),
                    'in',
                    int(bad.sum()),
                    'of',
                    bad.size,
                    'values;',
                    counts,
                )
    print('compare:', 'OK' if ret_flag == 0 else 'FAILED')
    return ret_flag
