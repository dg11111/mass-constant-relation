"""
轻子质量谱拟合
"""
import numpy as np
from scipy.optimize import minimize
from constants import MASS_ELECTRON, MASS_MUON, MASS_TAU


def lepton_mass_formula(n, params):
    """轻子质量公式: m_n = A * n^B * exp(C*n)"""
    A, B, C = params
    return A * n ** B * np.exp(C * n)


def fit_lepton_masses():
    """拟合轻子质量"""
    n_data = np.array([1, 2, 3])  # e, μ, τ
    m_data = np.array([MASS_ELECTRON, MASS_MUON, MASS_TAU])

    # 定义损失函数（对数误差）
    def loss(params):
        predictions = lepton_mass_formula(n_data, params)
        return np.sum((np.log(predictions) - np.log(m_data)) ** 2)

    # 初始猜测
    initial_guess = [1.0, 8.0, -0.7]
    result = minimize(loss, initial_guess, method='L-BFGS-B')

    if result.success:
        A, B, C = result.x
        print("\n" + "=" * 50)
        print("轻子质量谱拟合结果")
        print("=" * 50)
        print(f"最优参数: A = {A:.8f}, B = {B:.8f}, C = {C:.8f}")

        # 打印对比
        print(f"\n{'粒子':<6} {'实验值(MeV)':<15} {'拟合值(MeV)':<15} {'相对误差':<10}")
        print("-" * 50)
        for i, name in enumerate(['e', 'μ', 'τ']):
            m_exp = m_data[i]
            m_fit = lepton_mass_formula(n_data[i], result.x)
            error = (m_fit - m_exp) / m_exp * 100
            print(f"{name:<6} {m_exp:<15.10f} {m_fit:<15.10f} {error:>8.6f}%")

        return result.x
    else:
        print("拟合失败!")
        return None


if __name__ == "__main__":
    fit_lepton_masses()