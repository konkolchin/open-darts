from __future__ import annotations

import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter

from darts.tools.hdf5_tools import load_hdf5_to_dict

if TYPE_CHECKING:
    from darts.models.darts_model import DartsModel


@dataclass(frozen=True)
class WellProfilePlotSpec:
    """
    Description of one well-property profile plot.

    :param property_key: DataFrame column to plot.
    :param label: Optional x-axis label.
    :param output_name: Optional saved-file stem.
    :param component_index: Optional component index for vector-valued columns.
    :param offset: Offset added to values, for unit conversions.
    """

    property_key: str
    label: str | None = None
    output_name: str | None = None
    component_index: int | None = None
    offset: float = 0.0


def plot_well_property_profiles(
    well_name: str,
    coupled_model: DartsModel,
    reported_times: Sequence[float] | None = None,
    *,
    report_time_labels: Sequence[str] | None = None,
    properties: Sequence[str | WellProfilePlotSpec] | None = None,
    property_labels: dict[str, str] | None = None,
    property_offsets: dict[str, float] | None = None,
    output_dir: str | None = None,
    save_as: str | Sequence[str] | None = "pdf",
    show_plot: bool = True,
    cmap_name: str = "jet",
    figure_size: tuple[float, float] = (10, 6),
    marker_size: float = 4.0,
    line_width: float = 1.5,
    time_atol: float = 1e-10,
    time_rtol: float = 1e-8,
) -> dict[str, list[str]]:
    """
    Plot configurable property profiles for a DFM well.

    If reported_times is omitted, every available time in the well HDF5 file is
    plotted. If properties is omitted, scalar columns and numeric components of
    vector-valued columns in the well property DataFrame are plotted.

    :param well_name: Name of the well to plot.
    :param coupled_model: An initialized DartsModel with output files available.
    :param reported_times: Simulation times to plot, in days.
    :param report_time_labels: Optional legend labels for reported_times.
    :param properties: Optional property names or WellProfilePlotSpec objects.
                       Vector components can be requested as "column:index".
    :param property_labels: Optional x-axis labels keyed by property name or
                            "column:index" component key.
    :param property_offsets: Optional value offsets keyed by property name or
                             "column:index" component key.
    :param output_dir: Optional output directory. Defaults to a subfolder in the
                       model output folder.
    :param save_as: File extension or iterable of extensions to save. If None,
                    figures are not saved.
    :param show_plot: If True, display each figure.
    :param cmap_name: Matplotlib colormap name for logarithmically spaced report-step colors.
    :param time_atol: Absolute tolerance for matching requested times.
    :param time_rtol: Relative tolerance for matching requested times.
    :return: Dictionary mapping each plotted property key to saved figure paths.
    """
    well_props_file_path = os.path.join(
        coupled_model.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    data_frame = pd.read_pickle(well_props_file_path)

    h5_well_dict = load_hdf5_to_dict(coupled_model.well_filepath)
    simulated_time = np.asarray(h5_well_dict["dynamic"]["time"], dtype=float)
    report_indices, report_time_labels = _get_report_steps(
        simulated_time,
        reported_times,
        report_time_labels,
        time_atol,
        time_rtol,
    )

    well_geom = coupled_model.wells[well_name].geometry
    num_segments = well_geom.num_segments
    segment_depths = np.asarray(well_geom.z, dtype=float)

    plot_specs = _get_plot_specs(
        data_frame,
        properties,
        property_labels or {},
        property_offsets or {},
    )
    if len(plot_specs) == 0:
        raise ValueError("No plottable well properties were found.")

    output_dir = _prepare_output_dir(coupled_model, well_name, output_dir)
    saved_files = {}
    for spec in plot_specs:
        fig, ax = _plot_profile(
            data_frame,
            spec,
            report_indices,
            report_time_labels,
            num_segments,
            segment_depths,
            cmap_name,
            figure_size,
            marker_size,
            line_width,
        )
        saved_files[_get_spec_key(spec)] = _save_profile_figure(
            fig,
            output_dir,
            spec,
            save_as,
        )
        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


def _get_report_steps(
    simulated_time: np.ndarray,
    reported_times: Sequence[float] | None,
    report_time_labels: Sequence[str] | None,
    time_atol: float,
    time_rtol: float,
) -> tuple[list[int], list[str]]:
    if reported_times is None:
        report_indices = list(range(len(simulated_time)))
        selected_times = simulated_time
    else:
        selected_times = np.asarray(reported_times, dtype=float)
        if len(selected_times) == 0:
            raise ValueError("reported_times must contain at least one time value.")
        report_indices = []
        for requested_time in selected_times:
            matches = np.where(
                np.isclose(
                    simulated_time,
                    requested_time,
                    atol=time_atol,
                    rtol=time_rtol,
                )
            )[0]
            if len(matches) == 0:
                nearest_idx = int(np.argmin(np.abs(simulated_time - requested_time)))
                raise ValueError(
                    f"Requested time {requested_time:g} day was not found in "
                    f"{os.path.basename('well_data.h5')}. Nearest available time "
                    f"is {simulated_time[nearest_idx]:g} day."
                )
            report_indices.append(int(matches[0]))

    if report_time_labels is None:
        labels = [_format_time_label(time) for time in selected_times]
    else:
        labels = list(report_time_labels)
        if len(labels) != len(report_indices):
            raise ValueError(
                "report_time_labels must have the same length as reported_times."
            )

    return report_indices, labels


def _get_plot_specs(
    data_frame: pd.DataFrame,
    properties: Sequence[str | WellProfilePlotSpec] | None,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> list[WellProfilePlotSpec]:
    if properties is None:
        return _infer_plot_specs(data_frame, property_labels, property_offsets)

    specs = []
    for property_spec in properties:
        if isinstance(property_spec, WellProfilePlotSpec):
            specs.append(property_spec)
            continue

        property_key, component_index = _parse_property_name(property_spec)
        spec_key = _make_spec_key(property_key, component_index)
        specs.append(
            WellProfilePlotSpec(
                property_key=property_key,
                label=property_labels.get(spec_key, property_labels.get(property_key)),
                output_name=make_safe_filename(spec_key),
                component_index=component_index,
                offset=property_offsets.get(
                    spec_key,
                    property_offsets.get(property_key, 0.0),
                ),
            )
        )

    _validate_plot_specs(data_frame, specs)
    return specs


def _infer_plot_specs(
    data_frame: pd.DataFrame,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> list[WellProfilePlotSpec]:
    specs = []
    for column in data_frame.columns:
        sample = _get_first_valid_value(data_frame[column])
        if sample is None:
            if pd.api.types.is_numeric_dtype(data_frame[column]):
                specs.append(
                    _make_plot_spec(column, None, property_labels, property_offsets)
                )
            continue
        sample_array = np.asarray(sample)
        if sample_array.ndim == 0:
            specs.append(
                _make_plot_spec(column, None, property_labels, property_offsets)
            )
        elif sample_array.ndim == 1:
            for component_index in range(len(sample_array)):
                specs.append(
                    _make_plot_spec(
                        column,
                        component_index,
                        property_labels,
                        property_offsets,
                    )
                )

    return [
        spec
        for spec in specs
        if data_frame[spec.property_key].dropna().empty
        or _is_numeric_profile(data_frame, spec.property_key, spec.component_index)
    ]


def _make_plot_spec(
    property_key: str,
    component_index: int | None,
    property_labels: dict[str, str],
    property_offsets: dict[str, float],
) -> WellProfilePlotSpec:
    spec_key = _make_spec_key(property_key, component_index)
    return WellProfilePlotSpec(
        property_key=property_key,
        label=property_labels.get(spec_key, property_labels.get(property_key)),
        output_name=make_safe_filename(spec_key),
        component_index=component_index,
        offset=property_offsets.get(spec_key, property_offsets.get(property_key, 0.0)),
    )


def _validate_plot_specs(
    data_frame: pd.DataFrame,
    specs: Sequence[WellProfilePlotSpec],
) -> None:
    for spec in specs:
        if spec.property_key not in data_frame.columns:
            raise ValueError(
                f"Column '{spec.property_key}' was not found in well property data."
            )
        if data_frame[spec.property_key].dropna().empty:
            continue
        if not _is_numeric_profile(data_frame, spec.property_key, spec.component_index):
            raise ValueError(
                f"Column '{_get_spec_key(spec)}' is not numeric and cannot be plotted."
            )


def _plot_profile(
    data_frame: pd.DataFrame,
    spec: WellProfilePlotSpec,
    report_indices: Sequence[int],
    report_time_labels: Sequence[str],
    num_segments: int,
    segment_depths: np.ndarray,
    cmap_name: str,
    figure_size: tuple[float, float],
    marker_size: float,
    line_width: float,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figure_size)
    colors = _get_report_colors(len(report_indices), cmap_name)
    markers = ("o", "s", "d", "^", "v", "x", "*")
    line_styles = ("-", "--", "-.", ":")

    for idx, report_index in enumerate(report_indices):
        profile = _get_profile_values(data_frame, spec, report_index, num_segments)
        ax.plot(
            profile,
            segment_depths,
            marker=markers[idx % len(markers)],
            linestyle=line_styles[idx % len(line_styles)],
            color=colors[idx],
            label=report_time_labels[idx],
            markersize=marker_size,
            linewidth=line_width,
        )

    ax.set_xlabel(_get_axis_label(spec), fontsize=16, labelpad=10)
    ax.set_ylabel("Well segment depth [m]", fontsize=16, labelpad=10)
    ax.tick_params(axis="both", labelsize=12)
    ax.invert_yaxis()
    ax.grid(linestyle="--")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
        ncol=1,
        title="Report steps",
        title_fontsize=12,
    ).get_frame().set_edgecolor("black")
    ax.legend_.set_frame_on(True)
    fig.tight_layout()
    return fig, ax


def _get_profile_values(
    data_frame: pd.DataFrame,
    spec: WellProfilePlotSpec,
    report_index: int,
    num_segments: int,
) -> np.ndarray:
    start = report_index * num_segments
    stop = start + num_segments
    values = data_frame[spec.property_key].iloc[start:stop]
    if len(values) != num_segments:
        raise ValueError(
            f"Column '{spec.property_key}' has no complete profile at time "
            f"index {report_index}."
        )

    profile = _line_property_values(
        data_frame.iloc[start:stop], spec.property_key, spec.component_index
    )
    return profile + spec.offset


def _line_property_phase(property_key):
    if property_key.startswith("rho"):
        return property_key[3:]
    if property_key.startswith("mu"):
        return property_key[2:]
    if (
        property_key.startswith("x")
        and property_key.endswith("_mass")
        and "_in_" in property_key
    ):
        return property_key[1:-5].rsplit("_in_", 1)[1]
    return None


def _line_property_values(data_frame, property_key, component_index=None):
    """
    Mask undefined phase properties without hiding meaningful zero values.

    Density and viscosity must be positive. Phase-specific properties require
    a present phase when saturation or density is available. Zero saturation,
    flow, and component fractions in present phases remain valid.
    """
    if property_key not in data_frame.columns:
        return np.full(len(data_frame), np.nan)
    series = data_frame[property_key]
    if component_index is None:
        values = series.to_numpy(dtype=float, copy=True)
    else:
        values = np.asarray(
            [
                np.asarray(value)[component_index] if np.asarray(value).ndim else np.nan
                for value in series
            ],
            dtype=float,
        )
    valid = np.isfinite(values)
    phase = _line_property_phase(property_key)
    if phase:
        saturation_key = f"s{phase}"
        density_key = f"rho{phase}"
        if saturation_key in data_frame.columns:
            saturation = data_frame[saturation_key].to_numpy(dtype=float)
            valid &= np.isfinite(saturation) & (saturation > 0)
        if density_key in data_frame.columns:
            density = data_frame[density_key].to_numpy(dtype=float)
            valid &= np.isfinite(density) & (density > 0)
        if property_key.startswith(("rho", "mu")):
            valid &= values > 0
    values[~valid] = np.nan
    return values


def _prepare_output_dir(
    coupled_model: DartsModel,
    well_name: str,
    output_dir: str | None,
) -> str:
    remove_existing = output_dir is None
    if output_dir is None:
        output_dir = os.path.join(
            coupled_model.output_folder,
            f"well_prop_profiles_{well_name}",
        )

    output_dir = os.path.normpath(output_dir)
    if remove_existing and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _save_profile_figure(
    fig: Figure,
    output_dir: str,
    spec: WellProfilePlotSpec,
    save_as: str | Sequence[str] | None,
) -> list[str]:
    if save_as is None:
        return []
    if isinstance(save_as, str):
        extensions = (save_as,)
    else:
        extensions = tuple(save_as)

    output_name = spec.output_name or make_safe_filename(_get_spec_key(spec))
    saved_files = []
    for extension in extensions:
        extension = extension.lstrip(".")
        file_path = os.path.join(output_dir, f"{output_name}_profiles.{extension}")
        fig.savefig(file_path)
        saved_files.append(file_path)
    return saved_files


def _parse_property_name(property_name: str) -> tuple[str, int | None]:
    if ":" not in property_name:
        return property_name, None

    property_key, component_index = property_name.rsplit(":", 1)
    try:
        return property_key, int(component_index)
    except ValueError as exc:
        raise ValueError(
            "Vector-valued properties must be requested as 'column:index'."
        ) from exc


def _get_first_valid_value(series: pd.Series) -> object | None:
    values = series.dropna()
    if values.empty:
        return None
    return values.iloc[0]


def _is_numeric_profile(
    data_frame: pd.DataFrame,
    property_key: str,
    component_index: int | None,
) -> bool:
    sample = _get_first_valid_value(data_frame[property_key])
    if sample is None:
        return False
    try:
        if component_index is None:
            float(sample)
        else:
            float(np.asarray(sample)[component_index])
    except (IndexError, TypeError, ValueError):
        return False
    return True


def _get_report_colors(
    num_reports: int, cmap_name: str
) -> list[tuple[float, float, float, float]]:
    cmap = mpl.colormaps[cmap_name]
    positions = np.arange(num_reports) / (num_reports - 1 if num_reports > 1 else 1)
    positions = np.log1p(10.0 * positions) / np.log1p(10.0)
    return [cmap(value) for value in positions]


def _get_axis_label(spec: WellProfilePlotSpec) -> str:
    if spec.label is not None:
        return spec.label
    return _get_spec_key(spec)


def _get_spec_key(spec: WellProfilePlotSpec) -> str:
    return _make_spec_key(spec.property_key, spec.component_index)


def _make_spec_key(property_key: str, component_index: int | None) -> str:
    if component_index is None:
        return property_key
    return f"{property_key}:{component_index}"


def _format_time_label(time_days: float) -> str:
    seconds = time_days * 24.0 * 60.0 * 60.0
    if np.isclose(seconds, 0.0):
        return "0 s"
    if seconds < 60.0:
        return f"{seconds:g} s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:g} min"
    hours = minutes / 60.0
    if hours < 24.0:
        return f"{hours:g} h"
    return f"{time_days:g} d"


def make_safe_filename(value: str) -> str:
    """
    Convert free-form text to a filesystem-safe filename stem.

    :param value: Input value.
    :return: Filename-safe value.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_well_1d_reservoir_line_graphs(
    well_name: str,
    coupled_model: DartsModel,
    reported_times: list,
    *,
    report_time_labels: list = None,
    reservoir_radius_max: float = None,
    phase_labels: dict = None,
    property_labels: dict = None,
    output_dir: str = None,
    save_as=("png"),
    show_plot: bool = False,
    legend_loc: str = "best",
    cmap_name: str = "jet",
    figure_size=(6, 5),
    marker_size: float = 4.0,
    line_width: float = 2.0,
    time_atol: float = 1e-10,
    time_rtol: float = 1e-8,
):
    """
    Save scalar property profiles through a DFM well and a 1D reservoir.

    Reservoir properties are evaluated directly from reservoir_solution.h5 using
    coupled_model.output.output_properties.

    :param well_name: DFM well name.
    :param coupled_model: Initialized DartsModel instance with output files available.
    :param reported_times: Simulation times to plot, in days.
    :param report_time_labels: Legend labels for reported_times. If omitted, labels are
                               generated from the time values.
    :param reservoir_radius_max: Optional maximum reservoir radial distance [m]
                                 to include in the plot. If omitted, the full 1D reservoir is plotted.
    :param phase_labels: Optional mapping from phase keys to plot labels, for example
                         {"L_a": r"Liquid $CO_2$", "L_b": "Aqueous phase"}.
    :param property_labels: Optional mapping from property keys to full x-axis labels.
                            This overrides the automatic label generated for each property.
    :param output_dir: Directory for saved figures. Defaults to a subfolder in model output.
    :param save_as: File extension or iterable of extensions to save.
    :param show_plot: If True, display the figure.
    :param legend_loc: Matplotlib legend location.
    :param cmap_name: Matplotlib colormap name for logarithmically spaced report-step colors.
    :param figure_size: Matplotlib figure size.
    :param marker_size: Marker size.
    :param line_width: Line width.
    :param time_atol: Absolute tolerance for matching requested times.
    :param time_rtol: Relative tolerance for matching requested times.

    :returns: Dictionary mapping each plotted property to its saved figure paths.
    """
    _validate_well_res_1d(coupled_model)
    reported_times = np.asarray(reported_times, dtype=float)
    if len(reported_times) == 0:
        raise ValueError("reported_times must contain at least one time value.")

    if report_time_labels is None:
        report_time_labels = [_format_time_label(time) for time in reported_times]
    elif len(report_time_labels) != len(reported_times):
        raise ValueError(
            "report_time_labels must have the same length as reported_times."
        )

    phase_labels = phase_labels or {}
    property_labels = property_labels or {}

    well_props_file_path = os.path.join(
        coupled_model.output_folder, f"dfm_well_props_{well_name}.pkl"
    )
    well_data_frame = pd.read_pickle(well_props_file_path)
    prop_names, skipped_props = _get_available_property_names(
        coupled_model, well_data_frame
    )
    if len(prop_names) == 0:
        raise ValueError(
            "No scalar well-reservoir properties are available for plotting."
        )
    if skipped_props:
        print(
            "Skipped unavailable well-reservoir properties: " + ", ".join(skipped_props)
        )

    h5_well_dict = load_hdf5_to_dict(coupled_model.well_filepath)
    well_times = h5_well_dict["dynamic"]["time"]
    well_time_indices = _get_matching_time_indices(
        well_times, reported_times, "well_data.h5", time_atol, time_rtol
    )

    requested_props = list(prop_names)
    available_props = {
        key
        for container in coupled_model.physics.property_containers.values()
        for key in container.output_props
    }
    for prop in prop_names:
        phase = _line_property_phase(prop)
        if phase:
            for companion in (f"s{phase}", f"rho{phase}"):
                if companion in available_props and companion not in requested_props:
                    requested_props.append(companion)
    reservoir_times, reservoir_props = coupled_model.output.output_properties(
        output_properties=requested_props
    )
    reservoir_time_indices = _get_matching_time_indices(
        reservoir_times,
        reported_times,
        "reservoir_solution.h5",
        time_atol,
        time_rtol,
    )

    well_geom = coupled_model.wells[well_name].geometry
    segment_depths = np.asarray(well_geom.z, dtype=float)
    num_segments = well_geom.num_segments

    reservoir_radial_distance_all = _get_well_res_radial_distance(coupled_model)
    reservoir_cell_indices = _get_well_res_radius_indices(
        reservoir_radial_distance_all, reservoir_radius_max
    )
    reservoir_radial_distance = reservoir_radial_distance_all[reservoir_cell_indices]
    reservoir_prop_matrices = {}
    for prop in requested_props:
        reservoir_prop_matrix = np.asarray(reservoir_props[prop], dtype=float)
        if reservoir_prop_matrix.shape[1] < reservoir_radial_distance_all.size:
            raise ValueError(
                f"Reservoir property '{prop}' has {reservoir_prop_matrix.shape[1]} "
                f"cells, but the 1D reservoir geometry has {reservoir_radial_distance_all.size} cells."
            )
        reservoir_prop_matrices[prop] = reservoir_prop_matrix

    reservoir_frame = pd.DataFrame(
        {prop: values.reshape(-1) for prop, values in reservoir_prop_matrices.items()}
    )
    for prop in prop_names:
        matrix = reservoir_prop_matrices[prop]
        reservoir_prop_matrices[prop] = _line_property_values(
            reservoir_frame, prop
        ).reshape(matrix.shape)

    output_dir = _prepare_well_res_output_dir(coupled_model, well_name, output_dir)

    saved_files = {}
    for prop_name in prop_names:
        fig, ax = _plot_well_res_profile(
            prop_name,
            well_data_frame,
            reservoir_prop_matrices[prop_name],
            well_time_indices,
            reservoir_time_indices,
            reported_times,
            report_time_labels,
            segment_depths,
            num_segments,
            reservoir_radial_distance,
            reservoir_cell_indices,
            phase_labels,
            property_labels,
            legend_loc,
            cmap_name,
            figure_size,
            marker_size,
            line_width,
        )
        saved_files[prop_name] = _save_well_res_profile_figure(
            fig, output_dir, prop_name, save_as
        )

        if show_plot:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


def _plot_well_res_profile(
    prop_name,
    well_data_frame,
    reservoir_prop_matrix,
    well_time_indices,
    reservoir_time_indices,
    reported_times,
    report_time_labels,
    segment_depths,
    num_segments,
    reservoir_radial_distance,
    reservoir_cell_indices,
    phase_labels,
    property_labels,
    legend_loc,
    cmap_name,
    figure_size,
    marker_size,
    line_width,
):
    fig, ax = plt.subplots(figsize=figure_size)
    y_r, _offset, _rmin, _rmax = _configure_stacked_y_axis_linear_log(
        ax,
        segment_depths,
        reservoir_radial_distance,
        gap_ratio=0.06,
        n_ticks_well=5,
        add_minor=True,
    )

    colors = _get_report_colors(len(reported_times), cmap_name)
    markers = ["o", "s", "d", "^", "v", "x", "*"]
    linestyles = ["-", "--", "-.", ":"]

    for idx, (well_time_idx, reservoir_time_idx) in enumerate(
        zip(well_time_indices, reservoir_time_indices, strict=False)
    ):
        color = colors[idx]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]

        well_profile = _get_well_profile(
            well_data_frame, prop_name, well_time_idx, num_segments
        )
        reservoir_profile = reservoir_prop_matrix[
            reservoir_time_idx, reservoir_cell_indices
        ]

        if prop_name == "temperature":
            well_profile = well_profile - 273.15
            reservoir_profile = reservoir_profile - 273.15

        ax.plot(
            well_profile,
            segment_depths,
            linestyle=linestyle,
            marker=marker,
            linewidth=line_width,
            markersize=marker_size,
            color=color,
            label=report_time_labels[idx],
        )
        ax.plot(
            reservoir_profile,
            y_r(reservoir_radial_distance),
            linestyle=linestyle,
            marker=marker,
            linewidth=line_width,
            markersize=marker_size,
            color=color,
        )

    ax.set_ylim(top=0)
    ax.set_xlabel(_get_property_axis_label(prop_name, phase_labels, property_labels))
    ax.tick_params(
        axis="y",
        which="both",
        labelleft=True,
        labelright=False,
        left=True,
        right=False,
        pad=3,
        direction="in",
    )
    ax.spines["right"].set_visible(True)
    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.6)

    ax.legend(
        fontsize=7,
        loc=legend_loc,
        title="Report steps",
        title_fontsize=10,
    ).get_frame().set_edgecolor("black")

    fig.tight_layout()

    return fig, ax


def _save_well_res_profile_figure(fig, output_dir, prop_name, save_as):
    if save_as is None:
        return []
    elif isinstance(save_as, str):
        save_as = (save_as,)

    file_base = os.path.join(
        output_dir, f"{make_safe_filename(prop_name)}_well_1d_reservoir_profile"
    )
    saved_files = []
    for extension in save_as:
        extension = extension.lstrip(".")
        file_path = f"{file_base}.{extension}"
        fig.savefig(file_path)
        saved_files.append(file_path)

    return saved_files


def _get_available_property_names(coupled_model, well_data_frame):
    candidate_props = []
    skipped_props = []

    _append_unique(candidate_props, coupled_model.physics.vars)
    for container in coupled_model.physics.property_containers.values():
        _append_unique(candidate_props, container.output_props.keys())

    prop_names = []
    for prop in candidate_props:
        if prop not in well_data_frame.columns:
            skipped_props.append(prop)
        elif _is_scalar_property_series(well_data_frame[prop]):
            prop_names.append(prop)
        else:
            skipped_props.append(prop)

    return prop_names, skipped_props


def _append_unique(items, new_items):
    for item in new_items:
        if item not in items:
            items.append(item)


def _is_scalar_property_series(series):
    values = series.dropna()
    if values.empty:
        return True

    value = values.iloc[0]
    return np.asarray(value).ndim == 0


def _prepare_well_res_output_dir(coupled_model, well_name, output_dir):
    remove_old_default_output = output_dir is None
    if output_dir is None:
        output_dir = os.path.join(
            coupled_model.output_folder, f"well_1d_reservoir_line_graphs_{well_name}"
        )

    output_dir = os.path.normpath(output_dir)
    if remove_old_default_output:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def _validate_well_res_1d(coupled_model):
    dx = coupled_model.reservoir.global_data["dx"]
    if dx.ndim != 3 or dx.shape[1:] != (1, 1):
        raise ValueError(
            f"Expected a 1D reservoir with dx shape (*, 1, 1), got {dx.shape}."
        )


def _get_well_res_radial_distance(coupled_model):
    reservoir_dx = np.asarray(coupled_model.reservoir.global_data["dx"], dtype=float)
    reservoir_radial_distance = np.cumsum(reservoir_dx.flatten())
    if np.any(reservoir_radial_distance <= 0.0):
        raise ValueError("Reservoir radial distances must be strictly positive.")
    return reservoir_radial_distance


def _get_well_res_radius_indices(reservoir_radial_distance, reservoir_radius_max):
    if reservoir_radius_max is None:
        return np.arange(reservoir_radial_distance.size)

    reservoir_radius_max = float(reservoir_radius_max)
    if not np.isfinite(reservoir_radius_max) or reservoir_radius_max <= 0.0:
        raise ValueError("reservoir_radius_max must be a positive finite value.")

    indices = np.where(reservoir_radial_distance <= reservoir_radius_max)[0]
    if len(indices) < 2:
        if reservoir_radial_distance.size < 2:
            raise ValueError("At least two reservoir cells are needed for plotting.")
        raise ValueError(
            "reservoir_radius_max must include at least two reservoir cells. "
            f"Use a value >= {reservoir_radial_distance[1]:g} m, or omit "
            "reservoir_radius_max."
        )
    return indices


def _get_matching_time_indices(times, requested_times, source_name, atol, rtol):
    times = np.asarray(times, dtype=float)
    indices = []
    for requested_time in requested_times:
        matches = np.where(np.isclose(times, requested_time, atol=atol, rtol=rtol))[0]
        if len(matches) == 0:
            nearest_idx = int(np.argmin(np.abs(times - requested_time)))
            raise ValueError(
                f"Requested time {requested_time:g} day was not found in {source_name}. "
                f"Nearest available time is {times[nearest_idx]:g} day."
            )
        indices.append(int(matches[0]))
    return indices


def _get_well_profile(data_frame, prop_name, time_idx, num_segments):
    start = time_idx * num_segments
    stop = (time_idx + 1) * num_segments
    profile = data_frame[prop_name].iloc[start:stop].to_numpy()
    if len(profile) != num_segments:
        raise ValueError(
            f"Well property '{prop_name}' has no complete profile at time index {time_idx}."
        )
    try:
        return _line_property_values(data_frame.iloc[start:stop], prop_name)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Property '{prop_name}' is not a scalar profile and cannot be plotted by this function."
        ) from exc


def _get_property_axis_label(prop_name, phase_labels, property_labels):
    if prop_name in property_labels:
        return property_labels[prop_name]
    if prop_name == "pressure":
        return "Pressure [bar]"
    if prop_name == "temperature":
        return r"Temperature [$^\circ$C]"

    if prop_name.startswith("rho") and len(prop_name) > 3:
        phase = prop_name[3:]
        return rf"{_get_phase_label(phase, phase_labels)} density [kg/m$^3$]"
    if prop_name.startswith("mu") and len(prop_name) > 2:
        phase = prop_name[2:]
        return f"{_get_phase_label(phase, phase_labels)} viscosity [cP]"
    if prop_name.startswith("s") and len(prop_name) > 1:
        phase = prop_name[1:]
        return f"{_get_phase_label(phase, phase_labels)} saturation [-]"

    phase_mass_fraction = _parse_phase_mass_fraction(prop_name)
    if phase_mass_fraction is not None:
        component, phase = phase_mass_fraction
        return (
            f"{component} mass fraction in {_get_phase_label(phase, phase_labels)} [-]"
        )

    return prop_name


def _parse_phase_mass_fraction(prop_name):
    if not (prop_name.startswith("x") and prop_name.endswith("_mass")):
        return None
    inner = prop_name[1:-5]
    if "_in_" not in inner:
        return None
    component, phase = inner.split("_in_", 1)
    return component, phase


def _get_phase_label(phase, phase_labels):
    return phase_labels.get(phase, phase)


def _configure_stacked_y_axis_linear_log(
    ax,
    segment_depths,
    reservoir_radial_distance,
    *,
    gap_ratio=0.04,
    res_height=None,
    n_ticks_well=5,
    add_minor=True,
):
    """
    Build one y-axis that is linear for the well and logarithmic for the reservoir.
    Returns a mapper y_r(r) for reservoir radii.
    """
    z = np.asarray(segment_depths, float)
    r = np.asarray(reservoir_radial_distance, float)

    z0, z1 = float(z[0]), float(z[-1])
    z_span = abs(z1 - z0)
    gap = gap_ratio * z_span
    h_res = z_span if res_height is None else float(res_height)
    offset = z1 + gap

    rmin, rmax = float(np.min(r)), float(np.max(r))
    if not (rmin > 0 and rmax > rmin):
        raise ValueError("Reservoir radii must satisfy 0 < rmin < rmax.")

    log_base = 10
    logb = lambda x: np.log(x) / np.log(log_base)
    lmin, lmax = logb(rmin), logb(rmax)

    def y_r(rv):
        rv = np.clip(np.asarray(rv, float), rmin, rmax)
        return offset + h_res * (logb(rv) - lmin) / (lmax - lmin)

    ticks_w = np.linspace(z0, z1, n_ticks_well)
    labels_w = [f"{tick:g}" for tick in ticks_w]

    k0, k1 = int(np.floor(lmin)), int(np.ceil(lmax))
    decades = log_base ** np.arange(k0, k1 + 1, dtype=float)
    decades = decades[(decades >= rmin) & (decades <= rmax)]

    major_vals = np.unique(np.r_[rmin, decades])
    major_y = y_r(major_vals)

    def is_decade(value):
        k = round(logb(value))
        return np.isclose(value, log_base**k, rtol=0, atol=1e-12)

    def format_log_label(value):
        if is_decade(value) and not np.isclose(value, rmin):
            k = int(round(logb(value)))
            return rf"$\mathrm{{10}}^{{{k}}}$"
        return f"{value:g}"

    major_labels = [format_log_label(value) for value in major_vals]

    if add_minor:
        minors = []
        for k in range(k0, k1):
            base = log_base**k
            minors.extend(base * np.arange(2, log_base, dtype=float))
        minors = np.array(minors, float)
        minors = minors[(minors > rmin) & (minors < rmax)]
        ax.set_yticks(np.r_[ticks_w, major_y, y_r(minors)])
        ax.set_yticklabels(labels_w + major_labels + [""] * len(minors))
    else:
        ax.set_yticks(np.r_[ticks_w, major_y])
        ax.set_yticklabels(labels_w + major_labels)

    ax.set_ylim(z0, offset + h_res)
    ax.invert_yaxis()
    ax.axhline(offset - gap * 0.5, ls="--", lw=1)

    def ydata_to_axes(yval):
        return ax.transAxes.inverted().transform(ax.transData.transform((0, yval)))[1]

    y_well_mid_data = 0.5 * (z0 + z1)
    y_res_mid_data = y_r(np.sqrt(rmin * rmax))
    left_offset = -0.08

    ax.text(
        left_offset,
        ydata_to_axes(y_well_mid_data),
        r"$z_{\mathrm{well}}\,[\mathrm{m}]$ (linear)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        clip_on=False,
    )
    ax.text(
        left_offset,
        ydata_to_axes(y_res_mid_data),
        r"$r_\mathrm{reservoir}\,[\mathrm{m}]$ (logarithmic)",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="right",
        clip_on=False,
    )

    return y_r, offset, rmin, rmax
