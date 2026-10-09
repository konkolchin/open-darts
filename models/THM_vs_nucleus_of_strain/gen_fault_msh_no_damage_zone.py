import gmsh
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DOMAIN_W = 20000.0  # default x extent [m]; the fault crosses mid-depth at x = DOMAIN_W / 2
DOMAIN_H = 20000.0  # default y extent [m] (extrusion length)
LAYER_DY = 500.0    # default extrusion layer length along y [m]


DIP_DEGREES = 60.0  # default fault dip, measured from the horizontal

RESERVOIR_THICKNESS = 300.0  # default reservoir thickness [m] (= a + b below)
FAULT_OFFSET = 200.0         # default vertical juxtaposition/throw [m] (= b - a below)
DEPTH_SHIFT = 0.0            # default extra reservoir depth [m]; see the parameter below
MESH_LC = 400.0              # default far-field characteristic mesh size [m]


def gen_fault_msh_no_damage_zone(fault=False, mesh_dir='no_damage_zone', mesh_root='results',
                                 dip_degrees=DIP_DEGREES,
                                 domain_w=DOMAIN_W, domain_h=DOMAIN_H, layer_dy=LAYER_DY,
                                 reservoir_thickness=RESERVOIR_THICKNESS, fault_offset=FAULT_OFFSET,
                                 depth_shift=DEPTH_SHIFT, mesh_lc=MESH_LC, *, bnd_tags, matrix_tags, fault_tag):
    """
    Generate the simulation mesh (fault=False) or its fault-tagged twin (fault=True).

    :param dip_degrees: fault dip measured from the horizontal, as in
        gen_fault_msh.generate_3d_fault_mesh. The fault deepens towards -x, which is
        the negative phi used below; pass a positive angle. 0 and 180 degrees are a
        horizontal plane and have no fault trace, 90 is vertical.
    :param domain_w: x extent [m]; the fault crosses mid-depth at x = domain_w / 2.
    :param domain_h: y extent [m] (extrusion length).
    :param layer_dy: extrusion layer length along y [m].
    :param reservoir_thickness: reservoir thickness [m], the same on both sides of the fault.
    :param fault_offset: vertical juxtaposition/throw of the reservoir across the fault [m];
        0 removes the offset (the reservoir sits at the same depth on both sides).
    :param depth_shift: extra depth of the reservoir and the fault [m]. The domain grows by
        the same amount at its base, so the shallowest boundary stays at z = 0: that face
        carries a traction-free mechanical BC (reservoir.py set_boundary_conditions), i.e.
        it is the ground surface, and moving it down would leave the overburden above it
        unaccounted for - the deeper reservoir would then see the same stress as before.
    :param bnd_tags: physical tags of the outer boundary faces, keys 'BND_X-', 'BND_X+', 'BND_Y-',
        'BND_Y+', 'BND_Z-', 'BND_Z+' (idata.mesh.bnd_tags), required.
    :param matrix_tags: physical tags of the volumes (reservoir, overburden, underburden) (idata.mesh.matrix_tags), required.
    :param fault_tag: physical tag of the fault surfaces of the fault=True mesh (idata.mesh.fault_tag), required.
    :param mesh_lc: far-field characteristic mesh size [m]. The fault trace and the wells
        are refined to 0.1 of it and the reservoir boundaries to 0.5, so lowering it
        refines everything, the fault included. The cell count grows roughly with its
        square: the 2D section is extruded in layers of fixed layer_dy.
    """
    gmsh.initialize()
    # redirect gmsh's own Info:/Warning: output to a .log file (written below) instead of
    # the console: General.Terminal=0 stops it printing, logger.start() keeps capturing it
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.logger.start()
    gmsh.model.add("test_3D_fault")
    geo = gmsh.model.geo

    # ---- Parameters ----
    W = domain_w         # x extent
    H = domain_h         # y extent (extrusion length)
    # depth_shift deepens the reservoir/fault (D) and the domain base (Z2) together, so the
    # rock below the reservoir keeps its 2500 m and the free surface stays at Z1 = 0
    T = 5000.0 + depth_shift  # total thickness (Z2 - Z1)
    D = 2500.0 + depth_shift  # reservoir and fault mid-depth
    Z1 = 0.0             # top    = 0    (shallowest z, the free surface)
    Z2 = T               # bottom = 5000 + depth_shift (deepest z)
    # a, b: inner/outer reservoir-boundary half-thicknesses from D, assigned asymmetrically
    # to the domain's left/right edges below - that asymmetry is what juxtaposes the
    # reservoir across the fault. a+b = reservoir_thickness (same both sides), b-a =
    # fault_offset (the throw); fault_offset=0 (a=b) removes the offset entirely.
    a = (reservoir_thickness - fault_offset) / 2.0
    b = (reservoir_thickness + fault_offset) / 2.0
    # Fault half-length beyond the reservoir. This SETS THE FAULT Z EXTENT:
    # the buried fault spans z in [D - Lplus, D + Lplus], symmetric about D.
    Lplus = b + 200.0
    # dip from the horizontal, negative so the fault deepens towards -x. tan(phi) is
    # the trace slope dz/dx and is what places the fault points below; a horizontal
    # plane has no trace, and at exactly 90 degrees tan() overflows instead of
    # returning infinity, so both are rejected rather than silently mis-meshed.
    if abs(math.sin(math.radians(dip_degrees))) < 1.0e-8:
        raise ValueError('dip_degrees must not be 0 or 180 degrees')
    if abs(abs(dip_degrees) - 90.0) < 1.0e-8:
        raise ValueError('dip_degrees of exactly 90 is degenerate here (tan overflows); '
                         'use 89.9 for a near-vertical fault')
    phi = -abs(dip_degrees) * math.pi / 180.0
    lc = mesh_lc     # far-field characteristic size; the two multipliers below refine from it
    mult = 0.5       # reservoir-boundary points
    mult1 = 0.1      # fault trace and well points
    tol = 1e-6
    n_extrude = int(round(H / layer_dy))  # layers along y

    fault_x_at = lambda z: W / 2.0 + (z - D) / math.tan(phi)
    fault_x_range = sorted([fault_x_at(D - Lplus), fault_x_at(D + Lplus)])
    print(f"reservoir (left,  x=0): top={D - a:g} bottom={D + b:g} m")
    print(f"reservoir (right, x={W:g}): top={D - b:g} bottom={D + a:g} m")
    print(f"reservoir thickness={reservoir_thickness:g} m, fault offset={fault_offset:g} m")
    print(f"fault dip={abs(dip_degrees):g} deg, "
          f"z range=[{D - Lplus:g}, {D + Lplus:g}] m, "
          f"x range=[{fault_x_range[0]:g}, {fault_x_range[1]:g}] m")

    # --------- 2D section in the y = 0 plane (x-z), extruded in y ----------
    # Outer box points (x from 0 to W, z from Z1=top to Z2=bottom)
    p1 = geo.addPoint(0, 0, Z2, lc, tag=1)
    p2 = geo.addPoint(W, 0, Z2, lc, tag=2)
    p3 = geo.addPoint(W, 0, D + a, mult * lc, tag=3)
    p4 = geo.addPoint(W, 0, D - b, mult * lc, tag=4)
    p5 = geo.addPoint(W, 0, Z1, lc, tag=5)
    p6 = geo.addPoint(0, 0, Z1, lc, tag=6)
    p7 = geo.addPoint(0, 0, D - a, mult * lc, tag=7)
    p8 = geo.addPoint(0, 0, D + b, mult * lc, tag=8)

    # Inner fault-related points. p19/p20 are the buried fault tips at D +- Lplus.
    p15 = geo.addPoint(W / 2 + b / math.tan(phi), 0, D + b, mult1 * lc, tag=15)
    p16 = geo.addPoint(W / 2 + a / math.tan(phi), 0, D + a, mult1 * lc, tag=16)
    p17 = geo.addPoint(W / 2 - a / math.tan(phi), 0, D - a, mult1 * lc, tag=17)
    p18 = geo.addPoint(W / 2 - b / math.tan(phi), 0, D - b, mult1 * lc, tag=18)
    p19 = geo.addPoint(W / 2 - Lplus / math.tan(phi), 0, D - Lplus, mult1 * lc, tag=19)  # shallow tip
    p20 = geo.addPoint(W / 2 + Lplus / math.tan(phi), 0, D + Lplus, mult1 * lc, tag=20)  # deep tip

    # Lines
    l1 = geo.addLine(p1, p2, tag=1)
    l2 = geo.addLine(p2, p3, tag=2)
    l3 = geo.addLine(p3, p4, tag=3)
    l4 = geo.addLine(p4, p5, tag=4)
    l5 = geo.addLine(p5, p6, tag=5)
    l6 = geo.addLine(p6, p7, tag=6)
    l7 = geo.addLine(p7, p8, tag=7)
    l8 = geo.addLine(p8, p1, tag=8)

    l9 = geo.addLine(p3, p16, tag=9)
    l10 = geo.addLine(p4, p18, tag=10)
    l11 = geo.addLine(p8, p15, tag=11)
    l12 = geo.addLine(p7, p17, tag=12)
    l13 = geo.addLine(p18, p17, tag=13)
    l14 = geo.addLine(p17, p16, tag=14)
    l15 = geo.addLine(p16, p15, tag=15)
    l16 = geo.addLine(p15, p20, tag=16)   # fault extension into deep (underburden)
    l17 = geo.addLine(p18, p19, tag=17)   # fault extension into shallow (overburden)

    # Curve loops. cl5/cl6 are degenerate ([line, -line]) so the fault tip
    # lines l16/l17 get embedded INSIDE surfaces s1/s4 (buried fault tips).
    cl1 = geo.addCurveLoop([l1, l2, l9, l15, -l11, l8], tag=1)
    cl2 = geo.addCurveLoop([l3, l10, l13, l14, -l9], tag=2)
    cl3 = geo.addCurveLoop([-l12, l7, l11, -l15, -l14], tag=3)
    cl4 = geo.addCurveLoop([l4, l5, l6, l12, -l13, -l10], tag=4)
    cl5 = geo.addCurveLoop([l16, -l16], tag=5)
    cl6 = geo.addCurveLoop([l17, -l17], tag=6)

    s1 = geo.addPlaneSurface([cl1, -cl5], tag=1)   # deep  -> underburden
    s2 = geo.addPlaneSurface([cl2], tag=2)         # reservoir (deep side of fault)
    s3 = geo.addPlaneSurface([cl3], tag=3)         # reservoir (shallow side of fault)
    s4 = geo.addPlaneSurface([cl4, -cl6], tag=4)   # shallow -> overburden

    geo.synchronize()

    # --------- Extrude the section in +y with n_extrude layers ----------
    surfaces_to_extrude = [s1, s2, s3, s4]
    surface_extrusion_map = {}
    for surf in surfaces_to_extrude:
        extruded = geo.extrude([(2, surf)], 0, H, 0, [n_extrude], recombine=True)
        surface_extrusion_map[surf] = extruded
    geo.synchronize()

    all_surfaces = gmsh.model.getEntities(2)

    # ------------------------------------------------------------
    # Physical group tags
    # ------------------------------------------------------------
    LEFT = bnd_tags['BND_X-']     # x = 0
    RIGHT = bnd_tags['BND_X+']    # x = W
    FRONT = bnd_tags['BND_Y-']    # y = 0
    BACK = bnd_tags['BND_Y+']     # y = H
    TOP = bnd_tags['BND_Z-']      # z = Z1 (shallowest)
    BOT = bnd_tags['BND_Z+']      # z = Z2 (deepest)

    FAULT = fault_tag
    RESERVOIR, OVERBURDEN, UNDERBURDEN = matrix_tags

    # ------------------------------------------------------------
    # Tag outer boundary faces by bounding box (a surface belongs to a
    # boundary plane only if it is flat against that plane).
    # ------------------------------------------------------------
    left_surfaces, right_surfaces = [], []
    front_surfaces, back_surfaces = [], []
    top_surfaces, bottom_surfaces = [], []

    for dim, tag in all_surfaces:
        x_min, y_min, z_min, x_max, y_max, z_max = gmsh.model.getBoundingBox(dim, tag)
        if abs(x_min) < tol and abs(x_max) < tol:
            left_surfaces.append(tag)
        elif abs(x_min - W) < tol and abs(x_max - W) < tol:
            right_surfaces.append(tag)
        elif abs(y_min) < tol and abs(y_max) < tol:
            front_surfaces.append(tag)
        elif abs(y_min - H) < tol and abs(y_max - H) < tol:
            back_surfaces.append(tag)
        elif abs(z_min - Z1) < tol and abs(z_max - Z1) < tol:
            top_surfaces.append(tag)
        elif abs(z_min - Z2) < tol and abs(z_max - Z2) < tol:
            bottom_surfaces.append(tag)

    gmsh.model.addPhysicalGroup(2, left_surfaces, tag=LEFT, name="LEFT")
    gmsh.model.addPhysicalGroup(2, right_surfaces, tag=RIGHT, name="RIGHT")
    gmsh.model.addPhysicalGroup(2, front_surfaces, tag=FRONT, name="FRONT")
    gmsh.model.addPhysicalGroup(2, back_surfaces, tag=BACK, name="BACK")
    gmsh.model.addPhysicalGroup(2, top_surfaces, tag=TOP, name="TOP")
    gmsh.model.addPhysicalGroup(2, bottom_surfaces, tag=BOT, name="BOT")

    # ------------------------------------------------------------
    # Identify internal reservoir horizontal surfaces (constant z at the
    # damage-zone boundaries) so they are NOT mistaken for the fault.
    # ------------------------------------------------------------
    boundary_surfaces = set(left_surfaces + right_surfaces + front_surfaces +
                            back_surfaces + top_surfaces + bottom_surfaces)
    expected_z_levels = [D + a, D - a, D + b, D - b]
    reservoir_hz_surfaces = []
    for dim, tag in all_surfaces:
        if tag in boundary_surfaces:
            continue
        _, _, z_min, _, _, z_max = gmsh.model.getBoundingBox(dim, tag)
        if abs(z_min - z_max) < tol and any(abs(z_min - z) < tol for z in expected_z_levels):
            reservoir_hz_surfaces.append(tag)

    # Fault surfaces = internal, non-boundary, non-horizontal-reservoir surfaces.
    fault_surfaces = [tag for dim, tag in all_surfaces
                      if tag not in boundary_surfaces and tag not in reservoir_hz_surfaces]
    if fault and fault_surfaces:
        gmsh.model.addPhysicalGroup(2, fault_surfaces, tag=FAULT, name="FAULT")

    # ------------------------------------------------------------
    # Volume physical groups (robust: read volumes from the extrusion map).
    #   s1 -> deep   (underburden),  s4 -> shallow (overburden)
    #   s2, s3       -> reservoir
    # ------------------------------------------------------------
    def vol_of(surf):
        for e in surface_extrusion_map[surf]:
            if e[0] == 3:
                return e[1]
        return None

    under_vol = vol_of(s1)
    over_vol = vol_of(s4)
    res_vols = [vol_of(s2), vol_of(s3)]
    gmsh.model.addPhysicalGroup(3, res_vols, tag=RESERVOIR, name="RESERVOIR")
    gmsh.model.addPhysicalGroup(3, [over_vol], tag=OVERBURDEN, name="OVERBURDEN")
    gmsh.model.addPhysicalGroup(3, [under_vol], tag=UNDERBURDEN, name="UNDERBURDEN")

    print(f"LEFT={left_surfaces} RIGHT={right_surfaces} "
          f"FRONT={front_surfaces} BACK={back_surfaces} "
          f"TOP={top_surfaces} BOT={bottom_surfaces}")
    print(f"FAULT={fault_surfaces} (excluded reservoir hz={reservoir_hz_surfaces})")
    print(f"RESERVOIR={res_vols} OVERBURDEN={over_vol} UNDERBURDEN={under_vol}")

    # ------------------------------------------------------------
    # Vertical well centerlines + local refinement (kept away from the
    # fault, which sits near x = W/2).
    # ------------------------------------------------------------
    if False:
        try:
            wells = [
                (0.225 * W, 0.18 * H),
                (0.75 * W, 0.80 * H),
            ]
            well_lines = []
            for (wx, wy) in wells:
                p_bot = geo.addPoint(wx, wy, Z2, mult1 * lc)
                p_top = geo.addPoint(wx, wy, Z1, mult1 * lc)
                well_lines.append(geo.addLine(p_bot, p_top))
    
            geo.synchronize()
            vol_tags = [v for (dim, v) in gmsh.model.getEntities(3)]
            for l in well_lines:
                gmsh.model.mesh.embed(1, [l], 3, vol_tags)
    
            thr_fields = []
            for l in well_lines:
                fdist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fdist, "CurvesList", [l])
                gmsh.model.mesh.field.setNumber(fdist, "Sampling", 10000)
    
                fthr = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(fthr, "InField", fdist)
                gmsh.model.mesh.field.setNumber(fthr, "SizeMin", max(1.0, 0.05 * lc))
                gmsh.model.mesh.field.setNumber(fthr, "SizeMax", 1e12)
                gmsh.model.mesh.field.setNumber(fthr, "DistMin", 0.08)
                gmsh.model.mesh.field.setNumber(fthr, "DistMax", 10.0)
                gmsh.model.mesh.field.setNumber(fthr, "StopAtDistMax", 1)
                thr_fields.append(fthr)
    
            f_all = thr_fields[0] if len(thr_fields) == 1 else gmsh.model.mesh.field.add("Min")
            if len(thr_fields) > 1:
                gmsh.model.mesh.field.setNumbers(f_all, "FieldsList", thr_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(f_all)
            gmsh.option.setNumber("Mesh.MeshSizeFromFields", 1)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        except Exception as e:
            print(f"[WARN] Well refinement setup failed: {e}")

    mesh_filename = "mesh.msh"
    if fault:
        mesh_filename = "mesh_fault.msh"
    # absolute, so the mesh lands next to this script whatever the cwd is
    mesh_filename = os.path.join(BASE_DIR, mesh_root, mesh_dir, mesh_filename)
    os.makedirs(os.path.dirname(mesh_filename), exist_ok=True)
    log_filename = mesh_filename[:-len('.msh')] + '.log'

    # ------------------------------------------------------------
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.1)
    #gmsh.write("test_3d_fault_shifted.geo_unrolled")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()

    gmsh.write(mesh_filename)

    with open(log_filename, 'w') as f:
        f.write('\n'.join(gmsh.logger.get()) + '\n')
    gmsh.logger.stop()
    print('Gmsh output is written to the file', log_filename)

    gmsh.finalize()
    print('mesh generation is completed')


if __name__ == '__main__':
    # mesh.msh        : simulation mesh (no fault physical group)
    # mesh_fault.msh  : same geometry with the fault surfaces tagged FAULT,
    #                   used for post-processing (see fault.py)
    from cases.base import input_data_base
    from cases.homog import homog
    mesh = homog().mesh
    for fault in (False, True):
        gen_fault_msh_no_damage_zone(fault=fault, bnd_tags=mesh.bnd_tags, matrix_tags=mesh.matrix_tags, fault_tag=mesh.fault_tag)

