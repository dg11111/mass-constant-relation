"""
计算并验证质子质量公式
"""
import numpy as np
from constants import ALPHA, MASS_ELECTRON, MASS_PROTON


def calculate_proton_mass_ratio(alpha):
    """计算理论质子-电子质量比"""
    term1 = 3 * alpha ** (-1 / 2)
    term2 = 4 * np.pi * alpha ** (-1)
    term3 = 3 * alpha ** (-2 / 3)
    return term1 + term2 + term3, (term1, term2, term3)


def verify_formula():
    """验证公式精度"""
    mu_theory, terms = calculate_proton_mass_ratio(ALPHA)
    mu_experiment = MASS_PROTON / MASS_ELECTRON

    error_abs = mu_theory - mu_experiment
    error_rel = error_abs / mu_experiment * 100  # 百分比

    print("=" * 50)
    print("质子质量公式验证")
    print("=" * 50)
    print(f"理论值: μ_theory = {mu_theory:.10f}")
    print(f"实验值: μ_exp    = {mu_experiment:.10f}")
    print(f"绝对误差: Δμ = {error_abs:.8f}")
    print(f"相对误差: δ = {error_rel:.6f}%")
    print("\n各项贡献:")
    print(f"  项1 (3α^(-1/2)): {terms[0]:.6f}")
    print(f"  项2 (4πα^(-1)) : {terms[1]:.6f}")
    print(f"  项3 (3α^(-2/3)): {terms[2]:.6f}")

    return mu_theory, mu_experiment, error_rel


if __name__ == "__main__":
    verify_formula()