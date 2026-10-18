import math

import numpy as np
from numba import njit


@njit(cache=True, error_model="numpy")
def _average_phase_properties(
    saturation: np.ndarray,
    density: np.ndarray,
    viscosity: np.ndarray,
    composition: np.ndarray,
    liquid: bool,
) -> tuple:
    """
    Average one phase's segment properties at the internal pipe interfaces.

    For N segments, return N - 1 interface values. When both neighboring
    segments contain the phase, density and viscosity use arithmetic means;
    composition is weighted by density times saturation. When only one side
    contains the phase, copy that side's properties.

    Liquid presence is determined by positive density times saturation; if
    neither side has liquid, return zeros. Gas presence is determined by
    nonzero saturation; if neither side has gas, copy the right side.

    :param saturation: Segment phase saturations, shape (N,).
    :param density: Segment phase densities [kg/m3], shape (N,).
    :param viscosity: Segment phase viscosities [Pa.s], shape (N,).
    :param composition: Segment component mass fractions, shape (N, nc_fl).
    :param liquid: Whether to use the liquid presence and zero-weight rules.
    :return: Interface density, viscosity, and component mass-fraction arrays,
        with shapes (N - 1,), (N - 1,), and (N - 1, nc_fl), respectively.
    """
    n = len(saturation) - 1
    rho = np.zeros(n)
    mu = np.zeros(n)
    x = np.zeros((n, composition.shape[1]))
    for i in range(n):
        left_weight = density[i] * saturation[i]
        right_weight = density[i + 1] * saturation[i + 1]
        total_weight = left_weight + right_weight
        if liquid:
            from_right = left_weight == 0 and right_weight > 0
            from_left = right_weight == 0 and left_weight > 0
            average = total_weight > 0
        else:
            from_right = saturation[i] == 0
            from_left = saturation[i + 1] == 0
            average = True
        if from_right:
            rho[i], mu[i] = density[i + 1], viscosity[i + 1]
            x[i] = composition[i + 1]
        elif from_left:
            rho[i], mu[i] = density[i], viscosity[i]
            x[i] = composition[i]
        elif average:
            rho[i] = (density[i] + density[i + 1]) / 2
            mu[i] = (viscosity[i] + viscosity[i + 1]) / 2
            for j in range(composition.shape[1]):
                x[i, j] = (
                    composition[i, j] * density[i] * saturation[i]
                    + composition[i + 1, j] * density[i + 1] * saturation[i + 1]
                ) / total_weight
    return rho, mu, x


@njit(cache=True, error_model="numpy")
def _colebrook_fanning(reynolds: np.ndarray, relative_roughness: float) -> np.ndarray:
    """
    Solve the Fanning Colebrook equation in inverse square-root coordinates.

    With u = 1 / sqrt(f_F), solve
    u + 4 * log10(relative_roughness / 3.7065 + 1.2613 * u / Re) = 0.
    Here f_F is the Fanning factor, equal to one quarter of the Darcy factor.

    Newton iterations start from f_F = 0.005. A nonpositive proposed u is
    replaced by half the current u to keep the logarithm argument positive.
    Stop when the relative change in u is at most 1e-13, or after 30 iterations.
    Return the final estimate without a convergence flag; the caller determines
    which estimates to accept and which interfaces require a scalar fallback.

    :param reynolds: One-dimensional turbulent Reynolds numbers, Re > 2400.
    :param relative_roughness: Wall roughness divided by internal pipe diameter;
        must satisfy 0 <= relative_roughness < 3.7065 for this solver.
    :return: Dimensionless Fanning friction factors, one per Reynolds number.
    """
    result = np.empty_like(reynolds)
    for i in range(len(reynolds)):
        inverse_root = 1.0 / math.sqrt(0.005)
        coefficient = 1.2613 / reynolds[i]
        for _ in range(30):
            argument = relative_roughness / 3.7065 + coefficient * inverse_root
            residual = inverse_root + 4.0 * math.log10(argument)
            derivative = 1.0 + 4.0 * coefficient / (math.log(10.0) * argument)
            step = residual / derivative
            candidate = inverse_root - step
            if candidate <= 0:
                candidate = inverse_root / 2
            if abs(candidate - inverse_root) <= 1e-13 * inverse_root:
                inverse_root = candidate
                break
            inverse_root = candidate
        result[i] = 1.0 / inverse_root**2
    return result


@njit(cache=True, error_model="numpy")
def _phase_velocity_kernel(
    segment_saturations: tuple,
    face_properties: tuple,
    coefficients: tuple,
    derivatives: tuple,
    differentiate: bool,
) -> tuple:
    """
    Evaluate gas and combined-liquid velocities and optional local OBL derivatives.

    Positive velocity at interface i points from segment i to segment i + 1.
    A phase cannot flow from a segment where its saturation is zero: both its
    velocity and derivative row are set to zero for such an upstream segment.
    The profile parameter C00 and drift velocity vD0 are held fixed when
    differentiating; their derivatives are not included.

    :param segment_saturations: Tuple (sG, sL) of segment saturation arrays,
        each with N entries; sL combines the mobile liquid phases.
    :param face_properties: Tuple (sG_face, sL_face, rhoG_face, rhoL_face), each
        with N - 1 entries; densities are in kg/m3.
    :param coefficients: Tuple (C00, rhoM_vM, rhoM_adjusted_face, vD0) of
        interface arrays: dimensionless profile parameter, mixture mass flux
        [kg/m2/s], adjusted mixture density [kg/m3], and drift velocity [m/s].
    :param derivatives: Tuple of derivatives for sG_face, sL_face, rhoG_face,
        rhoL_face, rhoM_vM, and rhoM_adjusted_face, in that order. Each array
        has shape (N - 1, 2 * n_vars), with left-segment variables followed by
        right-segment variables. When differentiation is disabled, pass arrays
        with shape (N - 1, 0).
    :param differentiate: Whether to compute the local OBL velocity derivatives.
    :return: Tuple (vG, vL, vG_der, vL_der). Velocities are interface arrays in
        m/s; derivatives are per primary variable and have shape
        (N - 1, 2 * n_vars), or (N - 1, 0) when differentiation is disabled.
    """
    sG, sL = segment_saturations
    sG_face, sL_face, rhoG_face, rhoL_face = face_properties
    C00, rhoM_vM, rhoM_adjusted_face, vD0 = coefficients
    (
        sG_face_der,
        sL_face_der,
        rhoG_face_der,
        rhoL_face_der,
        rhoM_vM_der,
        rhoM_adjusted_face_der,
    ) = derivatives
    num_interfaces = len(sG_face)
    vG_der = np.empty((num_interfaces, 0))
    vL_der = np.empty((num_interfaces, 0))
    if differentiate:
        vG_der = np.zeros((num_interfaces, derivatives[0].shape[1]))
        vL_der = np.zeros((num_interfaces, derivatives[0].shape[1]))

    vG = np.zeros(num_interfaces)
    for i in range(num_interfaces):
        if sG_face[i] != 0:
            vG[i] = (
                C00[i] * rhoM_vM[i] / rhoM_adjusted_face[i]
                + rhoL_face[i] * vD0[i] / rhoM_adjusted_face[i]
            )

            if differentiate:
                for j in range(vG_der.shape[1]):
                    vG_der[i, j] = C00[i] * (
                        rhoM_vM_der[i, j] * rhoM_adjusted_face[i]
                        - rhoM_adjusted_face_der[i, j] * rhoM_vM[i]
                    ) / (rhoM_adjusted_face[i] ** 2) + vD0[i] * (
                        rhoL_face_der[i, j] * rhoM_adjusted_face[i]
                        - rhoM_adjusted_face_der[i, j] * rhoL_face[i]
                    ) / (rhoM_adjusted_face[i] ** 2)

    vL = np.zeros(num_interfaces)
    for i in range(num_interfaces):
        if sL_face[i] > 0:
            vL[i] = (1 - C00[i] * sG_face[i]) * rhoM_vM[i] / (
                sL_face[i] * rhoM_adjusted_face[i]
            ) - sG_face[i] * rhoG_face[i] * vD0[i] / (
                sL_face[i] * rhoM_adjusted_face[i]
            )

            if differentiate:
                for j in range(vL_der.shape[1]):
                    vL_der[i, j] = (
                        (
                            -C00[i] * sG_face_der[i, j] * rhoM_vM[i]
                            + (1 - C00[i] * sG_face[i]) * rhoM_vM_der[i, j]
                        )
                        * sL_face[i]
                        * rhoM_adjusted_face[i]
                        - (
                            sL_face_der[i, j] * rhoM_adjusted_face[i]
                            + sL_face[i] * rhoM_adjusted_face_der[i, j]
                        )
                        * (1 - C00[i] * sG_face[i])
                        * rhoM_vM[i]
                    ) / ((sL_face[i] * rhoM_adjusted_face[i]) ** 2) - (
                        (
                            vD0[i]
                            * (
                                sG_face_der[i, j] * rhoG_face[i]
                                + sG_face[i] * rhoG_face_der[i, j]
                            )
                        )
                        * (sL_face[i] * rhoM_adjusted_face[i])
                        - (
                            sL_face_der[i, j] * rhoM_adjusted_face[i]
                            + sL_face[i] * rhoM_adjusted_face_der[i, j]
                        )
                        * sG_face[i]
                        * rhoG_face[i]
                        * vD0[i]
                    ) / ((sL_face[i] * rhoM_adjusted_face[i]) ** 2)

    for i in range(num_interfaces):
        if vG[i] > 0 and sG[i] == 0:
            vG[i] = 0
            if differentiate:
                vG_der[i, :] = 0
        elif vG[i] < 0 and sG[i + 1] == 0:
            vG[i] = 0
            if differentiate:
                vG_der[i, :] = 0

        if vL[i] > 0 and sL[i] == 0:
            vL[i] = 0
            if differentiate:
                vL_der[i, :] = 0
        elif vL[i] < 0 and sL[i + 1] == 0:
            vL[i] = 0
            if differentiate:
                vL_der[i, :] = 0
    return vG, vL, vG_der, vL_der
