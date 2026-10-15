import importlib
import inspect
import os

from cases.base import (
    input_data_struct_like,
    check_rock_props,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# the struct-like rect_NXxNYxNZ meshes have no builder of their own: without a prefix they are
# HM (base_HM), with this prefix THM (base_THM). The results folder already names the physics,
# so the prefix is left out of it (get_output_directory).
THM = "thm_"
# cases/base.py holds the building blocks the cases are made of, not cases themselves
HELPER_MODULES = ('base',)


def discover_cases():
    """Collect the case builders defined in cases/*.py, keyed by their function name.

    A case is a public function that can be called without arguments (damage_zone has
    keyword arguments with defaults); the helpers in cases/base.py are not collected, and
    helpers in the other modules are private (_name). The case itself fixes its physics
    through base_HM or base_THM. Adding a case is therefore writing the function - nothing
    to register.

    Functions imported into a module are skipped (__module__ decides who owns a name), so
    heter importing homog does not claim it twice.
    """
    builders = {}
    for filename in sorted(os.listdir(os.path.join(BASE_DIR, 'cases'))):
        name, ext = os.path.splitext(filename)
        if ext != '.py' or name.startswith('_') or name in HELPER_MODULES:
            continue
        module = importlib.import_module('cases.' + name)
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn_name.startswith('_') or fn.__module__ != module.__name__:
                continue
            params = inspect.signature(fn).parameters.values()
            if any(p.default is inspect.Parameter.empty for p in params):
                continue
            if fn_name in builders:
                raise ValueError('two cases named %s: %s and %s'
                                 % (fn_name, builders[fn_name].__module__, module.__name__))
            builders[fn_name] = fn
    return builders


# A case is selected by the name of the function that builds it
CASE_BUILDERS = discover_cases()


def get_output_directory(model_folder, physics_type, wells_type):
    """Output folder of a run, also used to locate its reference solution.

    Lives here rather than in main so set_input_data can name the mesh folder after it -
    main imports it from here. The thm_ prefix of a struct-like case is dropped: the physics
    is already the first part of the folder name.
    """
    name = os.path.basename(model_folder)
    if name.startswith(THM):
        name = name[len(THM):]
    return os.path.join('results', physics_type + '_' + wells_type + '_' + name)


def get_case_output_directory(case):
    """Output folder of a case, with the physics and wells the case runs with."""
    idata = set_input_data(case)
    return get_output_directory(case, idata.other.physics_type, idata.other.wells_type)


def set_input_data(case: str):
    """
    Build the input data of a case; its physics and wells are idata.other.physics_type
    and idata.other.wells_type. The rock properties are printed by run() (print_idata).
    """
    case_ = os.path.basename(case) # without meshes/ part
    builder = CASE_BUILDERS.get(case_)
    if builder is not None:
        input_data = builder()
    else:
        # damage_zone mesh variants and the struct-like rect_NXxNYxNZ cases are named after
        # their mesh, so their names cannot be listed in CASE_BUILDERS
        name = case_
        thermal = name.startswith(THM)
        if thermal:
            name = name[len(THM):]
        if name.startswith("damage_zone_mesh_") and not thermal:
            input_data = CASE_BUILDERS["damage_zone"](mesh_dir=name)
        elif name.startswith("rect_"):
            input_data = input_data_struct_like(model_folder=name, thermal=thermal)
        else:
            raise ValueError("unknown case '%s': not a function in cases/*.py, nor [thm_]rect_NXxNYxNZ, "
                             "nor damage_zone_mesh_<size>m" % case_)

    # A generated mesh belongs with the run that generated it: put it in the run's own
    # output folder, next to its solution*.vtu. That also makes the mesh folder unique per
    # case, physics and well configuration without anyone naming it. A case shipping a
    # committed mesh (case_1, mesh_root 'meshes') keeps pointing at that file instead.
    if input_data.other.mesh_root == 'results':
        input_data.other.mesh_dir = os.path.basename(
            get_output_directory(case_, input_data.other.physics_type, input_data.other.wells_type))
        # the struct-like box mesh is written by the reservoir through gmsh, which fails
        # rather than creating the folder, and nothing else has made it at that point
        os.makedirs(os.path.join(BASE_DIR, input_data.other.mesh_root,
                                 input_data.other.mesh_dir), exist_ok=True)

    # every case is built through this function, so it is the one place where the rock
    # properties of all of them can be checked, with every override applied
    check_rock_props(input_data)

    return input_data
