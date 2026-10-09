from cases.base import input_data_struct_like, _set_reservoir_bounds

# Cases whose stress change can be compared with the uniaxial-strain formulas of the README:
# poroelastic (HM, injection into a closed reservoir) and thermoelastic (THM, injection, then the
# wells shut in (rate 0) so the pressure equilibrates while the cooled zone stays; run() switches the
# wells off at idata.other.shut_in_time).
# The names start with rect_ (thm_rect_ for THM), so the box mesh is generated on the fly


def _rect(mesh, thermal):
    idata = input_data_struct_like(model_folder=mesh, thermal=thermal)
    idata.other.rsv_xy = 10000.0  # [m] large reservoir width to mimic a laterally infinite reservoir
    _set_reservoir_bounds(idata)
    # high perm for faster pressure stabilization in the reservoir layers
    idata.rock.permx = idata.rock.permy = idata.rock.permz = 1000.0  # [mD]
    # almost impermeable to avoid pressure diffusion into under and over burden layers
    idata.rock.perm_non_rsv = 1e-6  # [mD]
    return idata


def _check_thermoelastic(mesh):
    # thermoelastic check (THM): the doublet injects cold water for 100 years; after 5 years shut in
    # the pressure change is ~0 and the cooled zone remains.
    idata = _rect(mesh, thermal=True)

    # wells 4 km apart along Y, so the cooled zone around the injector stays away from the producer
    # and from the mesh sides; Y = 0 lies between them, so XZ slices at Y = 0 miss the cooled zone
    # (dt_xy_contour.png shows it)
    idata.other.inj_well_coords[0:2] = [0., 2000.]
    idata.other.prod_well_coords[0:2] = [0., -2000.]

    # conduction 100x lower outside the reservoir (default 260 kJ/m/day/K everywhere), so the cooled
    # zone does not grow into the over- and underburden and stays a layer
    idata.rock.thermal_conductivity_non_rsv = 2.6  # [kJ/m/day/K]

    idata.other.shut_in_time = 100 * 365.25  # [days] wells switched to rate 0 at this time
    idata.other.runtime_years = 105
    return idata


def _check_poroelastic(mesh):
    # poroelastic check (HM): one injector in the reservoir centre under BHP control (initial
    # pressure + bhp_delta_p) pressurizes the closed reservoir for 10 years (base_HM); with 1000 mD inside and
    # an impermeable surrounding the pressure change is nearly uniform, and no temperature change
    # is involved, so the well is not shut in.
    idata = _rect(mesh, thermal=False)
    return idata


def thm_rect_17x17x15_check_analytical():
    return _check_thermoelastic('rect_17x17x15')


def rect_17x17x15_check_analytical():
    return _check_poroelastic('rect_17x17x15')


def thm_rect_41x41x66_check_analytical():
    return _check_thermoelastic('rect_41x41x66')


def rect_41x41x66_check_analytical():
    return _check_poroelastic('rect_41x41x66')


def thm_rect_71x71x66_check_analytical():
    return _check_thermoelastic('rect_71x71x66')


def rect_71x71x66_check_analytical():
    return _check_poroelastic('rect_71x71x66')
