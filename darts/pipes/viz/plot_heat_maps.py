import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator

from darts.tools.hdf5_tools import load_hdf5_to_dict


@dataclass(frozen=True)
class _Property:
    key: str
    name: str
    unit: str
    bins: str
    component: int | None = None
    offset: float = 0.0
    interfaces: bool = False
    mask_zero: bool = False
    fraction: bool = False
    title: str | None = None
    contour_label: str | None = None


def _properties(pc, n_mobile_phases, columns):
    yield _Property("pressure", "Pressure", "bar", "p")
    if "z" in columns:
        for index, component in enumerate(pc.components_name):
            yield _Property(
                "z",
                f"{component} overall mole fraction",
                "-",
                "comp",
                component=index,
                fraction=True,
                title=f"Profile of overall mole fraction of {component} along the wellbore over time",
            )
    if pc.thermal:
        yield _Property("temperature", "Temperature", "\u00b0C", "t", offset=-273.15)
    yield _Property("sG", "Gas volume fraction", "-", "s")
    liquids = ("L",) if n_mobile_phases == 2 else ()
    if n_mobile_phases == 3:
        liquids = ("L_a", "L_b")
        for phase in liquids:
            yield _Property(f"s{phase}", f"{phase} volume fraction", "-", "s")
    phases = ("G",) + liquids
    for phase in phases:
        for component in pc.components_name[: pc.nc_fl]:
            yield _Property(
                f"x{component}_in_{phase}_mass",
                f"{component} mass fraction in the {phase} phase",
                "-",
                "comp",
                fraction=True,
                title=f"Profile of {component} mass fraction in the {phase} phase along the wellbore over time",
                contour_label=f"{component} mass fraction in {phase} phase",
            )
    for prefix, quantity, unit, bins in (
        ("rho", "density", "kg/m$^3$", "rho"),
        ("mu", "viscosity", "cP", "mu"),
    ):
        for phase in phases:
            name = {"G": "Gas", "L": "Liquid"}.get(phase, phase)
            yield _Property(
                f"{prefix}{phase}",
                f"{name} {quantity}",
                unit,
                bins,
                mask_zero=True,
            )
    for phase, name in (("G", "Gas"), ("L", "Liquid")):
        if f"v{phase}" in columns:
            yield _Property(
                f"v{phase}",
                f"{name} velocity",
                "m/s",
                "v",
                interfaces=True,
                mask_zero=True,
            )
    for rate, unit in (
        ("molar", "kmol/s"),
        ("mass", "kg/s"),
        ("volumetric", "m$^3$/s"),
    ):
        for phase in pc.phases_name:
            key = f"phase_{rate}_rate_{phase}"
            if key in columns:
                name = {"G": "Gas", "L": "Liquid"}.get(phase, phase)
                yield _Property(key, f"{name} {rate} rate", unit, "v", interfaces=True)


def _property_matrix(data_frame, spec, num_segments, start, stop):
    values = data_frame[spec.key].iloc[start * num_segments : stop * num_segments]
    if spec.component is None:
        values = values.to_numpy(dtype=float)
    else:
        values = np.asarray(values.tolist(), dtype=float)[:, spec.component]
    matrix = values.reshape(stop - start, num_segments).T + spec.offset
    if spec.interfaces:
        matrix = matrix[:-1] / 86400.0
    matrix = np.ma.masked_invalid(matrix)
    if spec.mask_zero:
        matrix = np.ma.masked_equal(matrix, 0.0)
    return matrix


def _get_contour_levels(min_value, max_value, num_bins):
    min_value = float(np.ma.filled(min_value, np.nan))
    max_value = float(np.ma.filled(max_value, np.nan))
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        min_value, max_value = 0.0, 1.0
    elif min_value == max_value:
        delta = 1.0 if min_value == 0 else abs(min_value) * 0.05
        min_value -= delta
        max_value += delta
    return np.linspace(min_value, max_value, num_bins + 1)


def _coordinates(geometry, y_axis):
    if y_axis == "segment_index":
        return (
            np.arange(geometry.num_segments),
            np.arange(geometry.num_interfaces),
            "Segment index [-]",
            "Interface index [-]",
        )
    if y_axis == "segments_MD":
        return (
            geometry.z,
            geometry.z_interfaces,
            "Segment MD [meter]",
            "Interface MD [meter]",
        )
    if y_axis == "segments_TVD":
        return (
            geometry.TVD_segments,
            geometry.TVD_interfaces,
            "Segment TVD [meter]",
            "Interface TVD [meter]",
        )
    raise ValueError(
        "y_axis must be 'segments_MD', 'segments_TVD', or 'segment_index'."
    )


def _draw_heat_map(ax, x, y, matrix, spec, method, cmap_color, bins):
    if method == "pcolormesh":
        limits = (
            {"vmin": 0, "vmax": 1}
            if spec.key == "sG" or (spec.fraction and spec.component is None)
            else {}
        )
        return ax.pcolormesh(
            x, y, matrix, cmap=cmap_color, shading="auto", **limits
        ), {}
    lower, upper = matrix.min(), matrix.max()
    if spec.fraction and lower == upper:
        lower, upper = 0.0, 1.0
    levels = _get_contour_levels(lower, upper, bins[spec.bins])
    cmap = plt.get_cmap(cmap_color, bins[spec.bins])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    artist = ax.contourf(x, y, matrix, levels=levels, cmap=cmap, norm=norm)
    ax.contour(x, y, matrix, levels=levels, colors="k", linewidths=0.7)
    return artist, dict(boundaries=levels, ticks=levels, spacing="proportional")


def plot_heat_maps(
    well_name,
    coupled_model,
    *,
    method="contourf",
    min_ts_idx=0,
    max_ts_idx=None,
    x_axis="simulated_time",
    y_axis="segments_MD",
    cmap_color="jet",
    save_as="pdf",
    show_plot=True,
    font_size=14,
    with_title=True,
    y_axis_tick_interval=None,
    with_logarithmic_x_axis=False,
    bins=None,
):
    """
    Plot well-property heatmaps using filled contours or a color mesh.

    :param well_name: Well whose properties and geometry are plotted.
    :param coupled_model: Model providing well geometry and output file paths.
    :param method: Rendering method, either ``'contourf'`` or ``'pcolormesh'``.
    :param min_ts_idx: First included time-step index.
    :param max_ts_idx: Exclusive last time-step index; None selects all steps.
    :param x_axis: ``'simulated_time'`` in seconds or ``'time_step_index'``.
    :param y_axis: ``'segments_MD'``, ``'segments_TVD'``, or ``'segment_index'``.
    :param cmap_color: Matplotlib colormap name.
    :param save_as: Output image extension.
    :param show_plot: Whether to display each figure.
    :param font_size: Font size for labels, titles, and ticks.
    :param with_title: Whether to display figure titles.
    :param y_axis_tick_interval: Tick spacing; None uses 50 for contours and
                                automatic spacing for meshes (1 for indices).
    :param with_logarithmic_x_axis: Whether to use a logarithmic time axis.
    :param bins: Optional contour-bin overrides keyed by ``'p'``, ``'comp'``,
                 ``'t'``, ``'s'``, ``'rho'``, ``'mu'``, or ``'v'``; default 10.
    :return: None. Figures are saved in the method-specific heatmap folder.
    """
    if method not in ("contourf", "pcolormesh"):
        raise ValueError("method must be 'contourf' or 'pcolormesh'.")
    default_bins = dict.fromkeys(("p", "comp", "t", "s", "rho", "mu", "v"), 10)
    if bins is not None:
        if not isinstance(bins, Mapping):
            raise ValueError(
                "bins must be a mapping of property keys to positive integers."
            )
        if bins.keys() - default_bins.keys():
            raise ValueError("Unknown bins key; expected p, comp, t, s, rho, mu, or v.")
        default_bins.update(bins)
    bins = default_bins
    for key, count in bins.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
            raise ValueError(f"bins['{key}'] must be a positive integer.")
    if y_axis_tick_interval is not None and (
        not np.isfinite(y_axis_tick_interval) or y_axis_tick_interval <= 0
    ):
        raise ValueError("y_axis_tick_interval must be finite and positive.")
    if y_axis_tick_interval is None and method == "contourf":
        y_axis_tick_interval = 50.0
    well = coupled_model.wells[well_name]
    geometry = well.geometry
    data_frame = pd.read_pickle(
        os.path.join(
            coupled_model.output_folder,
            f"dfm_well_props_{well_name}.pkl",
        )
    )
    num_segments = geometry.num_segments
    if num_segments <= 0 or len(data_frame) % num_segments:
        raise ValueError("Well property rows must contain complete segment profiles.")
    num_ts = len(data_frame) // num_segments
    if max_ts_idx is None:
        max_ts_idx = num_ts
    if not 0 <= min_ts_idx < max_ts_idx <= num_ts:
        raise ValueError(f"Time step indices must satisfy 0 <= min < max <= {num_ts}.")
    if method == "contourf" and max_ts_idx - min_ts_idx < 2:
        raise ValueError("Contour heatmaps require at least two selected time steps.")
    if x_axis == "simulated_time":
        time = load_hdf5_to_dict(coupled_model.well_filepath)["dynamic"]["time"]
        x = np.asarray(time)[min_ts_idx:max_ts_idx] * 86400.0
        if len(x) != max_ts_idx - min_ts_idx:
            raise ValueError("HDF5 times do not cover the selected property profiles.")
        x_label = "Simulated time [second]"
    elif x_axis == "time_step_index":
        x = np.arange(min_ts_idx, max_ts_idx)
        x_label = "Time step [-]"
    else:
        raise ValueError("x_axis must be 'simulated_time' or 'time_step_index'.")
    segments, interfaces, segment_label, interface_label = _coordinates(
        geometry, y_axis
    )
    output_dir = os.path.join(
        coupled_model.output_folder, f"heat_maps_{method}_{well_name}"
    )
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    pc = coupled_model.physics.property_containers[0]
    for counter, spec in enumerate(
        _properties(pc, well.n_mobile_phases, data_frame.columns)
    ):
        matrix = _property_matrix(
            data_frame, spec, num_segments, min_ts_idx, max_ts_idx
        )
        y = interfaces if spec.interfaces else segments
        if len(y) != matrix.shape[0]:
            raise ValueError(f"Geometry does not match the rows of '{spec.key}'.")
        if (
            method == "contourf"
            and spec.key.startswith("phase_")
            and matrix.count() == 0
        ):
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            artist, colorbar_options = _draw_heat_map(
                ax,
                x,
                y,
                matrix,
                spec,
                method,
                cmap_color,
                bins,
            )
            cbar = fig.colorbar(artist, ax=ax, **colorbar_options)
            label = spec.contour_label if method == "contourf" else None
            cbar.set_label(f"{label or spec.name} [{spec.unit}]", fontsize=font_size)
            cbar.ax.tick_params(labelsize=font_size)
            interval = y_axis_tick_interval
            if interval is None and y_axis == "segment_index":
                interval = 1
            if interval is not None:
                ax.yaxis.set_major_locator(MultipleLocator(interval))
            ax.invert_yaxis()
            ax.set_xlabel(x_label, fontsize=font_size)
            ax.set_ylabel(
                interface_label if spec.interfaces else segment_label,
                fontsize=font_size,
            )
            if with_logarithmic_x_axis:
                ax.set_xscale("log")
            ax.tick_params(axis="both", labelsize=font_size)
            if with_title:
                title = (
                    spec.title or f"{spec.name} profile along the wellbore over time"
                )
                ax.set_title(title, fontsize=font_size, fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{counter}- {spec.name}.{save_as}"))
            if show_plot:
                plt.show()
        finally:
            plt.close(fig)
