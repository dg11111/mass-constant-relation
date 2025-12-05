"""
强子八重态系统拟合器
基于公式: m_hadron/m_e = I_eff × (3α^{-1/2} + 4πα^{-1} + 3α^{-2/3})
其中I_eff = p + qY + r[I(I+1)-Y²/4] + sQ + t(Q·I₃)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import json


@dataclass
class BaryonData:
    """重子数据类"""
    name: str  # 粒子名称
    symbol: str  # 符号
    quark_content: str  # 夸克成分
    mass_mev: float  # 质量 (MeV)
    charge: float  # 电荷 Q
    isospin: float  # 同位旋 I
    isospin_z: float  # 同位旋第三分量 I₃
    strangeness: int  # 奇异数 S
    baryon_number: int  # 重子数 B

    @property
    def hypercharge(self) -> float:
        """超荷 Y = B + S"""
        return self.baryon_number + self.strangeness

    @property
    def mass_ratio(self) -> float:
        """质量比 m/m_e"""
        return self.mass_mev / 0.51099895000


class BaryonOctetFitter:
    """重子八重态拟合器"""

    def __init__(self):
        """初始化，载入八重态数据"""
        self.baryons = self.load_baryon_data()
        self.alpha = 1.0 / 137.035999139
        self.m_e = 0.51099895000

        # 基础质量函数
        self.base_mass_ratio = (
                3.0 * self.alpha ** (-0.5) +
                4.0 * np.pi * self.alpha ** (-1.0) +
                3.0 * self.alpha ** (-2.0 / 3.0)
        )

    def load_baryon_data(self) -> List[BaryonData]:
        """载入重子八重态数据"""
        return [
            # 核子 (N)
            BaryonData("质子", "p", "uud", 938.27208816, 1.0, 0.5, 0.5, 0, 1),
            BaryonData("中子", "n", "udd", 939.565420, 0.0, 0.5, -0.5, 0, 1),

            # Λ粒子
            BaryonData("Λ零", "Λ⁰", "uds", 1115.683, 0.0, 0.0, 0.0, -1, 1),

            # Σ粒子
            BaryonData("Σ⁺", "Σ⁺", "uus", 1189.37, 1.0, 1.0, 1.0, -1, 1),
            BaryonData("Σ⁰", "Σ⁰", "uds", 1192.642, 0.0, 1.0, 0.0, -1, 1),
            BaryonData("Σ⁻", "Σ⁻", "dds", 1197.449, -1.0, 1.0, -1.0, -1, 1),

            # Ξ粒子
            BaryonData("Ξ⁰", "Ξ⁰", "uss", 1314.86, 0.0, 0.5, 0.5, -2, 1),
            BaryonData("Ξ⁻", "Ξ⁻", "dss", 1321.71, -1.0, 0.5, -0.5, -2, 1)
        ]

    def compute_ieff(self, params: Dict[str, float], baryon: BaryonData) -> float:
        """
        计算I_eff值

        I_eff = p + qY + r[I(I+1)-Y²/4] + sQ + t(Q·I₃)
        """
        p = params.get('p', 0.0)
        q = params.get('q', 0.0)
        r = params.get('r', 0.0)
        s = params.get('s', 0.0)
        t = params.get('t', 0.0)

        Y = baryon.hypercharge
        I = baryon.isospin
        Q = baryon.charge
        I3 = baryon.isospin_z

        # 计算各项
        term1 = p
        term2 = q * Y
        term3 = r * (I * (I + 1) - Y ** 2 / 4.0)
        term4 = s * Q
        term5 = t * (Q * I3)

        return term1 + term2 + term3 + term4 + term5

    def mass_function(self, params: Dict[str, float], baryon: BaryonData) -> float:
        """
        质量计算函数

        m/m_e = I_eff × F(α) + 修正项
        F(α) = 3α^{-1/2} + 4πα^{-1} + 3α^{-2/3}
        """
        ieff = self.compute_ieff(params, baryon)

        # 基础质量贡献
        base_mass = ieff * self.base_mass_ratio

        # 对带电粒子添加电磁修正
        em_correction = 0.0
        if abs(baryon.charge) > 0.5:  # 带电粒子
            em_correction = params.get('em_corr', 0.83)

        # 对奇异粒子添加奇异修正
        s_correction = 0.0
        if baryon.strangeness < 0:  # 含奇异夸克
            s_factor = abs(baryon.strangeness)
            s_correction = params.get('s_corr', 0.0) * s_factor

        return base_mass + em_correction + s_correction

    def fit_parameters(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        拟合I_eff参数

        Returns
        -------
        Tuple[Dict[str, float], Dict[str, float]]
            拟合参数和协方差矩阵
        """
        # 准备数据
        x_data = []
        y_data = []
        baryon_indices = []

        for i, baryon in enumerate(self.baryons):
            x_data.append({
                'baryon': baryon,
                'index': i
            })
            y_data.append(baryon.mass_ratio)
            baryon_indices.append(i)

        # 定义拟合函数
        def fit_func(x_array, p, q, r, s, t, em_corr, s_corr):
            """供curve_fit使用的拟合函数"""
            params = {
                'p': p, 'q': q, 'r': r, 's': s, 't': t,
                'em_corr': em_corr, 's_corr': s_corr
            }

            results = []
            for x in x_array:
                baryon_idx = int(x[0])
                baryon = self.baryons[baryon_idx]
                results.append(self.mass_function(params, baryon))

            return np.array(results)

        # 初始猜测值
        p0 = [3.0, 0.6, 0.5, 0.01, 0.01, 0.83, 0.1]

        # 准备x数据（包含重子索引）
        x_fit = np.array([[i] for i in range(len(self.baryons))])
        y_fit = np.array(y_data)

        # 执行拟合
        try:
            popt, pcov = curve_fit(
                fit_func, x_fit, y_fit,
                p0=p0,
                bounds=(
                    [2.5, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0],  # 下界
                    [3.5, 1.0, 1.0, 0.1, 0.1, 2.0, 0.5]  # 上界
                ),
                maxfev=10000
            )

            # 提取参数
            params = {
                'p': popt[0],
                'q': popt[1],
                'r': popt[2],
                's': popt[3],
                't': popt[4],
                'em_corr': popt[5],
                's_corr': popt[6]
            }

            # 计算误差
            perr = np.sqrt(np.diag(pcov))
            errors = {
                'p': perr[0],
                'q': perr[1],
                'r': perr[2],
                's': perr[3],
                't': perr[4],
                'em_corr': perr[5],
                's_corr': perr[6]
            }

            return params, errors

        except Exception as e:
            print(f"拟合失败: {e}")
            # 使用预设值
            default_params = {
                'p': 3.000, 'q': 0.618, 'r': 0.500,
                's': 0.007, 't': 0.010,
                'em_corr': 0.83, 's_corr': 0.12
            }
            return default_params, {}

    def calculate_predictions(self, params: Dict[str, float]) -> pd.DataFrame:
        """
        计算预测值与误差

        Returns
        -------
        pd.DataFrame
            包含所有重子的详细计算结果
        """
        results = []

        for baryon in self.baryons:
            # 计算预测值
            predicted_ratio = self.mass_function(params, baryon)
            predicted_mev = predicted_ratio * self.m_e

            # 实验值
            experimental_mev = baryon.mass_mev

            # 误差分析
            abs_error_mev = predicted_mev - experimental_mev
            rel_error_percent = (abs_error_mev / experimental_mev) * 100

            # 计算I_eff值
            ieff = self.compute_ieff(params, baryon)

            # 各贡献项
            contributions = {
                'base_term': self.base_mass_ratio,
                'ieff_value': ieff,
                'em_contrib': params.get('em_corr', 0.0) if abs(baryon.charge) > 0.5 else 0.0,
                's_contrib': params.get('s_corr', 0.0) * abs(baryon.strangeness) if baryon.strangeness < 0 else 0.0
            }

            results.append({
                '粒子': baryon.name,
                '符号': baryon.symbol,
                '夸克成分': baryon.quark_content,
                '实验质量_MeV': experimental_mev,
                '预测质量_MeV': predicted_mev,
                '绝对误差_MeV': abs_error_mev,
                '相对误差_%': rel_error_percent,
                'I_eff值': ieff,
                '电荷Q': baryon.charge,
                '超荷Y': baryon.hypercharge,
                '同位旋I': baryon.isospin,
                'I₃': baryon.isospin_z,
                '奇异数S': baryon.strangeness
            })

        return pd.DataFrame(results)

    def plot_results(self, df: pd.DataFrame, params: Dict[str, float],
                     save_path: str = "results/baryon_fit.png"):
        """
        绘制拟合结果图

        Parameters
        ----------
        df : pd.DataFrame
            计算结果数据框
        params : Dict[str, float]
            拟合参数
        save_path : str
            保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 质量对比图
        ax1 = axes[0, 0]
        particles = df['符号'].values
        exp_masses = df['实验质量_MeV'].values
        pred_masses = df['预测质量_MeV'].values

        x = np.arange(len(particles))
        width = 0.35

        ax1.bar(x - width / 2, exp_masses, width, label='实验值', alpha=0.8, color='skyblue')
        ax1.bar(x + width / 2, pred_masses, width, label='预测值', alpha=0.8, color='lightcoral')

        ax1.set_xlabel('粒子')
        ax1.set_ylabel('质量 (MeV)')
        ax1.set_title('重子八重态质量对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(particles, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 误差分布图
        ax2 = axes[0, 1]
        errors = df['相对误差_%'].values

        colors = ['red' if abs(e) > 0.1 else 'green' for e in errors]
        ax2.bar(particles, errors, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axhline(y=-0.1, color='orange', linestyle='--', linewidth=1, alpha=0.5)

        ax2.set_xlabel('粒子')
        ax2.set_ylabel('相对误差 (%)')
        ax2.set_title('质量预测误差分布')
        ax2.set_xticklabels(particles, rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. I_eff值与质量关系
        ax3 = axes[1, 0]
        ieff_values = df['I_eff值'].values

        ax3.scatter(ieff_values, exp_masses, s=100, alpha=0.7, label='实验值')
        ax3.scatter(ieff_values, pred_masses, s=100, alpha=0.7, marker='x', label='预测值')

        # 添加标签
        for i, particle in enumerate(particles):
            ax3.annotate(particle, (ieff_values[i], exp_masses[i]),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=8)

        ax3.set_xlabel('I_eff 值')
        ax3.set_ylabel('质量 (MeV)')
        ax3.set_title('I_eff 值与质量关系')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 参数贡献分析
        ax4 = axes[1, 1]
        param_names = ['p', 'q', 'r', 's', 't']
        param_values = [params.get(name, 0.0) for name in param_names]

        ax4.bar(param_names, param_values, alpha=0.7, color='teal')
        ax4.set_xlabel('参数')
        ax4.set_ylabel('参数值')
        ax4.set_title('I_eff 参数拟合结果')
        ax4.grid(True, alpha=0.3, axis='y')

        # 在图上添加参数值
        for i, v in enumerate(param_values):
            ax4.text(i, v + 0.01 * max(param_values), f'{v:.3f}',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"图表已保存至: {save_path}")

    def generate_report(self, params: Dict[str, float], errors: Dict[str, float],
                        df: pd.DataFrame) -> str:
        """
        生成详细报告

        Returns
        -------
        str
            报告文本
        """
        report = []
        report.append("=" * 80)
        report.append("重子八重态系统拟合报告")
        report.append("=" * 80)
        report.append("")

        # 1. 拟合参数
        report.append("1. I_eff 参数拟合结果:")
        report.append("   I_eff = p + qY + r[I(I+1)-Y²/4] + sQ + t(Q·I₃)")
        report.append("")

        for param in ['p', 'q', 'r', 's', 't']:
            value = params.get(param, 0.0)
            error = errors.get(param, 0.0)
            if error > 0:
                report.append(f"   {param} = {value:.6f} ± {error:.6f}")
            else:
                report.append(f"   {param} = {value:.6f}")

        report.append("")
        report.append(f"   电磁修正参数 em_corr = {params.get('em_corr', 0.0):.6f}")
        report.append(f"   奇异修正参数 s_corr = {params.get('s_corr', 0.0):.6f}")
        report.append("")

        # 2. 统计信息
        rel_errors = df['相对误差_%'].abs().values
        max_error = np.max(rel_errors)
        mean_error = np.mean(rel_errors)
        rms_error = np.sqrt(np.mean(rel_errors ** 2))

        report.append("2. 拟合质量统计:")
        report.append(f"   最大绝对误差: {df['绝对误差_MeV'].abs().max():.3f} MeV")
        report.append(f"   平均绝对误差: {df['绝对误差_MeV'].abs().mean():.3f} MeV")
        report.append(f"   最大相对误差: {max_error:.4f} %")
        report.append(f"   平均相对误差: {mean_error:.4f} %")
        report.append(f"   均方根误差: {rms_error:.4f} %")
        report.append("")

        # 3. 各粒子结果
        report.append("3. 各重子详细结果:")
        report.append("-" * 80)

        for _, row in df.iterrows():
            report.append(f"   {row['粒子']} ({row['符号']}, {row['夸克成分']}):")
            report.append(f"       实验质量: {row['实验质量_MeV']:.3f} MeV")
            report.append(f"       预测质量: {row['预测质量_MeV']:.3f} MeV")
            report.append(f"       绝对误差: {row['绝对误差_MeV']:.3f} MeV")
            report.append(f"       相对误差: {row['相对误差_%']:.4f} %")
            report.append(f"       I_eff值: {row['I_eff值']:.4f}")
            report.append("")

        # 4. 物理解释
        report.append("4. 物理意义解读:")
        report.append("   p (基准值): 三个u/d强小子的平均耦合强度")
        report.append("   q (超荷项): 奇异强小子的额外耦合增强")
        report.append("   r (同位旋项): 味道对称性破缺的程度")
        report.append("   s (电荷项): 电磁相互作用对凝聚层的调制")
        report.append("   t (电荷-同位旋项): 电荷分布对称性的精细效应")
        report.append("")
        report.append("   I_eff值反映了每个重子内部的'有效量子数'，")
        report.append("   它编码了夸克成分、电磁效应和对称性的综合影响。")

        return "\n".join(report)


def main():
    """主函数"""
    print("重子八重态系统拟合器启动...")
    print("-" * 50)

    # 创建拟合器
    fitter = BaryonOctetFitter()

    # 拟合参数
    print("正在拟合参数...")
    params, errors = fitter.fit_parameters()

    # 计算预测值
    df = fitter.calculate_predictions(params)

    # 生成报告
    report = fitter.generate_report(params, errors, df)
    print(report)

    # 绘制图表
    print("正在生成图表...")
    fitter.plot_results(df, params)

    # 保存结果
    output_dir = "results"
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 保存DataFrame
    df.to_csv(f"{output_dir}/baryon_predictions.csv", index=False, encoding='utf-8-sig')

    # 保存参数
    with open(f"{output_dir}/baryon_fit_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "parameters": params,
            "errors": errors,
            "base_mass_ratio": fitter.base_mass_ratio,
            "alpha": fitter.alpha
        }, f, indent=2, ensure_ascii=False)

    print("-" * 50)
    print("结果已保存至 results/ 目录:")
    print("1. baryon_predictions.csv - 详细计算结果")
    print("2. baryon_fit_params.json - 拟合参数")
    print("3. baryon_fit.png - 分析图表")


if __name__ == "__main__":
    main()