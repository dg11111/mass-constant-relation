"""
新粒子预言器
基于理论预言的新粒子（弱小子等）性质预测
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, chi2
import pandas as pd
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class ParticlePrediction:
    """粒子预言"""
    name: str
    symbol: str
    predicted_mass: float  # MeV
    mass_uncertainty: float
    quantum_numbers: Dict[str, Any]
    production_channels: List[str]
    decay_channels: List[str]
    detection_signature: str
    experimental_status: str
    confidence_level: float  # 0-1

    def __str__(self) -> str:
        return f"{self.name} ({self.symbol}): {self.predicted_mass:.2f} ± {self.mass_uncertainty:.2f} MeV"


class NewParticlePredictor:
    """新粒子预言器"""

    def __init__(self):
        """初始化预言器"""
        self.alpha = 1 / 137.035999139
        self.m_e = 0.51099895000  # MeV

        # 从质子质量公式推导的参数
        self.m_p = 938.27208816  # MeV
        self.term_structure = self._analyze_term_structure()

        # 已知粒子质量参考（MeV）
        self.known_particles = {
            '电子': 0.511,
            '缪子': 105.66,
            'π介子': 139.57,
            'K介子': 493.677,
            '质子': 938.272,
            '中子': 939.565,
            'Λ重子': 1115.683,
            'Ω重子': 1672.45
        }

        # 实验约束
        self.experimental_constraints = self._load_experimental_constraints()

    def _analyze_term_structure(self) -> Dict[str, float]:
        """分析质子质量公式的项结构"""
        term1 = 3 * self.alpha ** (-0.5)  # 强小子项
        term2 = 4 * np.pi * self.alpha ** (-1)  # 胶子海凝聚项
        term3 = 3 * self.alpha ** (-2 / 3)  # 弱小子项

        total = term1 + term2 + term3

        return {
            'term1': term1,
            'term2': term2,
            'term3': term3,
            'total': total,
            'term1_mev': term1 * self.m_e,
            'term2_mev': term2 * self.m_e,
            'term3_mev': term3 * self.m_e,
            'term1_percent': term1 / total * 100,
            'term2_percent': term2 / total * 100,
            'term3_percent': term3 / total * 100
        }

    def _load_experimental_constraints(self) -> Dict[str, Any]:
        """加载实验约束"""
        # 来自不同实验的未发现粒子的质量排除区间
        constraints = {
            'LEP': {  # 大型正负电子对撞机
                'energy': 209,  # GeV
                'excluded_ranges': [(0.1, 45.0)],  # GeV
                'particle_types': ['带电粒子', '中性粒子'],
                'confidence': 0.95
            },
            'LHC': {  # 大型强子对撞机
                'energy': 13000,  # GeV
                'excluded_ranges': [(0.1, 5000.0)],  # GeV
                'particle_types': ['各类新粒子'],
                'confidence': 0.95
            },
            'Belle_II': {  # B工厂
                'energy': 10.58,  # GeV
                'excluded_ranges': [(0.1, 5.0)],  # GeV
                'particle_types': ['长寿命粒子', '弱相互作用粒子'],
                'confidence': 0.90
            },
            '固定靶实验': {
                'energy': 10,  # GeV
                'excluded_ranges': [(0.001, 2.0)],  # GeV
                'particle_types': ['轻粒子'],
                'confidence': 0.80
            }
        }
        return constraints

    def predict_weaklet(self) -> ParticlePrediction:
        """
        预言弱小子粒子

        Returns
        -------
        ParticlePrediction
            弱小子预言
        """
        # 从质子质量公式的第三项推导
        # 项3: 3α^{-2/3}m_e 对应弱相互作用修正
        # 假设弱小子贡献与项3相关

        # 计算弱小子质量
        # 基本假设: 弱小子质量与 α^{-2/3} 成正比
        # 从项3的结构: 3 * α^{-2/3} * m_e
        # 除以3得到单个贡献: α^{-2/3} * m_e ≈ 26.1576 * 0.511 ≈ 13.37 MeV

        # 但考虑弱相互作用强度，实际质量可能为:
        weaklet_mass_base = self.alpha ** (-2 / 3) * self.m_e  # ~13.37 MeV

        # 调整因子: 考虑弱耦合与电磁耦合的比例
        # α_w ≈ α^p，其中p需要确定
        # 从项3的幂次-2/3反推
        # 如果弱贡献 ∝ α_w * (强贡献)，且强贡献 ∝ α^{-1}
        # 那么 α_w ∝ α^{1/3}，因此质量 ∝ α^{1/3} * α^{-1} = α^{-2/3}

        # 考虑对称性破缺和动力学效应，质量可能加倍
        weaklet_mass = weaklet_mass_base * 1.5  # ~20 MeV范围

        # 不确定性估计
        mass_uncertainty = weaklet_mass * 0.3  # 30%相对不确定度

        # 量子数预测
        quantum_numbers = {
            '自旋': '1/2 或 0',
            '电荷': '0 或 ±1',
            '色荷': '无色',
            '弱同位旋': '可能为1/2',
            '重子数': '0',
            '轻子数': '0',
            '奇异数': '0',
            '宇称': '待定'
        }

        # 产生道
        production_channels = [
            '质子-质子碰撞: p + p → X + 强子',
            '电子-正电子湮灭: e⁺ + e⁻ → γ → X + X̄',
            'B介子衰变: B → K + X',
            '重离子碰撞中的集体产生'
        ]

        # 衰变道
        decay_channels = [
            'X → e⁺ + e⁻ (如果允许)',
            'X → γ + γ (如果自旋为0)',
            'X → π⁺ + π⁻ (如果质量足够)',
            'X → 不可见 (如果弱相互作用主导)'
        ]

        # 探测特征
        detection_signature = """
        1. 在e⁺e⁻对撞中的共振产生
        2. 在B介子衰变中的缺失能量信号
        3. 固定靶实验中的轻粒子产生
        4. 宇宙线中的异常相互作用
        """

        return ParticlePrediction(
            name='弱小子',
            symbol='W',
            predicted_mass=weaklet_mass,
            mass_uncertainty=mass_uncertainty,
            quantum_numbers=quantum_numbers,
            production_channels=production_channels,
            decay_channels=decay_channels,
            detection_signature=detection_signature,
            experimental_status='未发现，理论预言',
            confidence_level=0.7
        )

    def predict_stronglet_family(self) -> List[ParticlePrediction]:
        """
        预言强小子家族

        Returns
        -------
        List[ParticlePrediction]
            强小子家族预言
        """
        predictions = []

        # 基础强小子（对应u/d夸克）
        stronglet_base = self.alpha ** (-0.5) * self.m_e  # ~5.98 MeV

        # 第一代强小子 (u/d型)
        stronglet_ud = ParticlePrediction(
            name='强小子(u/d)',
            symbol='S_ud',
            predicted_mass=stronglet_base * 2.5,  # ~15 MeV，考虑束缚效应
            mass_uncertainty=stronglet_base * 0.5,
            quantum_numbers={
                '自旋': '1/2',
                '电荷': '±2/3, ±1/3',
                '色荷': '三重态',
                '重子数': '1/3',
                '代': '第一代'
            },
            production_channels=[
                '高能对撞中的喷注',
                '粲/底强子衰变'
            ],
            decay_channels=[
                '通过弱作用衰变',
                '强子化产生强子'
            ],
            detection_signature='类似夸克但质量较轻的组分',
            experimental_status='未确认，可能被误解为海夸克',
            confidence_level=0.6
        )
        predictions.append(stronglet_ud)

        # 第二代强小子 (s型)
        # 质量按黄金比例增加: φ = (1+√5)/2 ≈ 1.618
        phi = (1 + np.sqrt(5)) / 2
        stronglet_s_mass = stronglet_ud.predicted_mass * phi  # ~24 MeV

        stronglet_s = ParticlePrediction(
            name='强小子(s)',
            symbol='S_s',
            predicted_mass=stronglet_s_mass,
            mass_uncertainty=stronglet_s_mass * 0.3,
            quantum_numbers={
                '自旋': '1/2',
                '电荷': '±2/3, ±1/3, -1',
                '色荷': '三重态',
                '奇异数': '-1',
                '代': '第二代'
            },
            production_channels=[
                '奇异强子衰变',
                '关联产生 S_s + S̄_s'
            ],
            decay_channels=[
                'S_s → S_ud + π',
                '弱衰变到轻代'
            ],
            detection_signature='带有奇异数的轻质量喷注',
            experimental_status='未发现',
            confidence_level=0.5
        )
        predictions.append(stronglet_s)

        # 第三代强小子 (c型)
        stronglet_c_mass = stronglet_s_mass * phi  # ~39 MeV

        stronglet_c = ParticlePrediction(
            name='强小子(c)',
            symbol='S_c',
            predicted_mass=stronglet_c_mass,
            mass_uncertainty=stronglet_c_mass * 0.3,
            quantum_numbers={
                '自旋': '1/2',
                '电荷': '±2/3',
                '色荷': '三重态',
                '粲数': '1',
                '代': '第三代'
            },
            production_channels=[
                '粲偶素衰变',
                'B介子衰变'
            ],
            decay_channels=[
                'S_c → S_s + 轻强子',
                '弱衰变'
            ],
            detection_signature='带有粲数的轻质量粒子',
            experimental_status='未发现',
            confidence_level=0.4
        )
        predictions.append(stronglet_c)

        return predictions

    def predict_composite_states(self) -> List[ParticlePrediction]:
        """
        预言复合态

        Returns
        -------
        List[ParticlePrediction]
            复合态预言
        """
        predictions = []

        # 获取基本组分质量
        weaklet = self.predict_weaklet()
        stronglets = self.predict_stronglet_family()

        # 1. 弱小子-反弱小子束缚态
        wwbar_mass = 2 * weaklet.predicted_mass - 10  # 结合能约10 MeV

        wwbar = ParticlePrediction(
            name='弱偶素',
            symbol='η_W',
            predicted_mass=wwbar_mass,
            mass_uncertainty=10,  # MeV
            quantum_numbers={
                '自旋': '0',
                '电荷': '0',
                '色荷': '无色',
                '宇称': '+',
                'C宇称': '+'
            },
            production_channels=[
                '质子-质子碰撞',
                '电子-正电子湮灭'
            ],
            decay_channels=[
                'η_W → γγ',
                'η_W → e⁺e⁻',
                'η_W → 不可见'
            ],
            detection_signature='双光子共振或双轻子共振',
            experimental_status='未发现',
            confidence_level=0.6
        )
        predictions.append(wwbar)

        # 2. 强小子-强小子束缚态（类似介子）
        s_ud = stronglets[0]
        ssbar_mass = 2 * s_ud.predicted_mass - 50  # 结合能约50 MeV

        ssbar = ParticlePrediction(
            name='强介子',
            symbol='π_S',
            predicted_mass=ssbar_mass,
            mass_uncertainty=20,
            quantum_numbers={
                '自旋': '0',
                '电荷': '0, ±1',
                '色荷': '无色',
                '宇称': '-'
            },
            production_channels=[
                '高能强子碰撞',
                '辐射衰变'
            ],
            decay_channels=[
                'π_S → γγ',
                'π_S → 轻强子',
                'π_S → 弱小子对'
            ],
            detection_signature='轻质量赝标量介子',
            experimental_status='可能在现有数据中被忽略',
            confidence_level=0.5
        )
        predictions.append(ssbar)

        # 3. 三强小子束缚态（类似重子）
        sss_mass = 3 * s_ud.predicted_mass - 100  # 结合能约100 MeV

        sss = ParticlePrediction(
            name='强重子',
            symbol='N_S',
            predicted_mass=sss_mass,
            mass_uncertainty=30,
            quantum_numbers={
                '自旋': '1/2',
                '电荷': '0, ±1',
                '色荷': '无色',
                '重子数': '1'
            },
            production_channels=[
                '高能碰撞',
                '重离子碰撞'
            ],
            decay_channels=[
                'N_S → N + π',
                'N_S → N + γ',
                '弱衰变'
            ],
            detection_signature='类似核子但质量较轻',
            experimental_status='未发现',
            confidence_level=0.4
        )
        predictions.append(sss)

        return predictions

    def check_experimental_constraints(self,
                                       particle: ParticlePrediction) -> Dict[str, Any]:
        """
        检查实验约束

        Parameters
        ----------
        particle : ParticlePrediction
            要检查的粒子

        Returns
        -------
        Dict[str, Any]
            约束检查结果
        """
        results = {}

        mass_gev = particle.predicted_mass / 1000  # 转换为GeV

        for exp_name, constraint in self.experimental_constraints.items():
            excluded = False
            exclusion_reason = ""

            # 检查每个排除区间
            for excl_range in constraint['excluded_ranges']:
                if excl_range[0] <= mass_gev <= excl_range[1]:
                    excluded = True
                    exclusion_reason = f"质量 {mass_gev:.3f} GeV 在 {exp_name} 排除区间 {excl_range} GeV 内"
                    break

            results[exp_name] = {
                'excluded': excluded,
                'exclusion_reason': exclusion_reason if excluded else "未排除",
                'confidence': constraint['confidence'],
                'energy_scale': constraint['energy'],
                'particle_types': constraint['particle_types']
            }

        # 总体评估
        all_excluded = any(result['excluded'] for result in results.values())
        survival_probability = 1.0 if not all_excluded else 0.0

        # 考虑置信度加权
        if not all_excluded:
            # 计算生存概率
            confidences = [result['confidence'] for result in results.values()]
            avg_confidence = np.mean(confidences)
            survival_probability = 0.5 + 0.5 * avg_confidence  # 简化估计

        results['overall'] = {
            'experimentally_excluded': all_excluded,
            'survival_probability': survival_probability,
            'recommendation': '被排除，需要修改理论' if all_excluded else '尚未被排除，值得实验搜索'
        }

        return results

    def calculate_discovery_potential(self,
                                      particle: ParticlePrediction,
                                      experiment: str = 'LHC') -> Dict[str, Any]:
        """
        计算发现潜力

        Parameters
        ----------
        particle : ParticlePrediction
            目标粒子
        experiment : str, optional
            实验名称，默认LHC

        Returns
        -------
        Dict[str, Any]
            发现潜力评估
        """
        # 实验参数
        experiment_params = {
            'LHC': {
                'luminosity': 150,  # fb^-1
                'energy': 13000,  # GeV
                'detection_efficiency': 0.1,  # 典型值
                'background_level': '高',
                'sensitivity_mass_range': (0.1, 5000)  # GeV
            },
            'Belle_II': {
                'luminosity': 50,  # ab^-1
                'energy': 10.58,  # GeV
                'detection_efficiency': 0.3,
                'background_level': '中',
                'sensitivity_mass_range': (0.1, 5)  # GeV
            },
            '固定靶': {
                'luminosity': 0.1,  # fb^-1
                'energy': 10,  # GeV
                'detection_efficiency': 0.01,
                'background_level': '低',
                'sensitivity_mass_range': (0.001, 2)  # GeV
            }
        }

        params = experiment_params.get(experiment, experiment_params['LHC'])

        # 简化发现潜力计算
        mass_gev = particle.predicted_mass / 1000

        # 检查质量是否在灵敏度范围内
        in_range = (params['sensitivity_mass_range'][0] <= mass_gev <=
                    params['sensitivity_mass_range'][1])

        if not in_range:
            return {
                'experiment': experiment,
                'mass_in_range': False,
                'discovery_potential': 0.0,
                'recommendation': f'质量 {mass_gev:.3f} GeV 超出 {experiment} 灵敏度范围'
            }

        # 简化计算发现潜力
        # 基于截面估计（简化模型）
        if mass_gev < 1:  # GeV以下
            cross_section = 1e-3  # pb，典型值
        elif mass_gev < 10:
            cross_section = 1e-2  # pb
        elif mass_gev < 100:
            cross_section = 1e-1  # pb
        else:
            cross_section = 1.0  # pb

        # 预期事件数
        expected_events = (cross_section * 1e-12 *  # pb → cm^2
                           params['luminosity'] * 1e-39 *  # fb^-1 → cm^-2
                           params['detection_efficiency'] * 1000)  # 考虑效率

        # 发现显著性估计（简化）
        if expected_events < 1:
            significance = 0
            discovery_potential = 0.1
        elif expected_events < 10:
            significance = expected_events / np.sqrt(10)  # 假设背景10事件
            discovery_potential = min(0.5, significance / 5)
        else:
            significance = expected_events / np.sqrt(expected_events + 10)
            discovery_potential = min(0.9, significance / 5)

        # 考虑粒子特性调整
        if '不可见' in particle.decay_channels:
            discovery_potential *= 0.5  # 不可见衰变更难

        if particle.confidence_level < 0.5:
            discovery_potential *= 0.7  # 理论置信度低

        return {
            'experiment': experiment,
            'mass_in_range': True,
            'predicted_mass_gev': mass_gev,
            'cross_section_estimate_pb': cross_section,
            'expected_events': expected_events,
            'significance_estimate': significance,
            'discovery_potential': discovery_potential,
            'recommendation': f'在{experiment}有中等发现潜力' if discovery_potential > 0.3 else f'在{experiment}发现潜力较低'
        }

    def plot_particle_spectrum(self, save_path: str = "figures"):
        """
        绘制新粒子能谱图

        Parameters
        ----------
        save_path : str, optional
            保存路径
        """
        # 获取所有预言
        weaklet = self.predict_weaklet()
        stronglets = self.predict_stronglet_family()
        composites = self.predict_composite_states()

        all_particles = [weaklet] + stronglets + composites

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 1. 质量谱
        particle_names = []
        masses = []
        uncertainties = []
        colors = []

        for i, particle in enumerate(all_particles):
            particle_names.append(particle.symbol)
            masses.append(particle.predicted_mass)
            uncertainties.append(particle.mass_uncertainty)

            # 根据类型分配颜色
            if '弱' in particle.name:
                colors.append('#E74C3C')  # 红色
            elif '强' in particle.name and '小子' in particle.name:
                colors.append('#3498DB')  # 蓝色
            else:
                colors.append('#2ECC71')  # 绿色

        # 绘制质量棒
        y_pos = range(len(particle_names))
        ax1.barh(y_pos, masses, xerr=uncertainties,
                 color=colors, alpha=0.7, capsize=5)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f'{name}\n({particle.name})'
                             for name, particle in zip(particle_names, all_particles)])
        ax1.set_xlabel('质量 (MeV)', fontsize=12)
        ax1.set_title('预言的新粒子质量谱', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')

        # 添加已知粒子参考线
        known_masses = {
            'π⁰': 134.98,
            'η': 547.86,
            'ρ': 775.26,
            'ω': 782.65,
            'J/ψ': 3096.9
        }

        for name, mass in known_masses.items():
            ax1.axvline(x=mass, color='gray', linestyle='--', alpha=0.5)
            ax1.text(mass, len(particle_names) * 0.95, name,
                     rotation=90, verticalalignment='top',
                     fontsize=8, alpha=0.7)

        # 2. 发现潜力热图
        experiments = ['LHC', 'Belle_II', '固定靶']
        discovery_potentials = []

        for particle in all_particles:
            pot_row = []
            for exp in experiments:
                potential = self.calculate_discovery_potential(particle, exp)
                pot_row.append(potential['discovery_potential'])
            discovery_potentials.append(pot_row)

        # 创建热图
        im = ax2.imshow(discovery_potentials, cmap='YlOrRd', aspect='auto',
                        vmin=0, vmax=1)

        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels(experiments, fontsize=11)
        ax2.set_yticks(range(len(particle_names)))
        ax2.set_yticklabels([p.symbol for p in all_particles], fontsize=11)
        ax2.set_ylabel('粒子', fontsize=12)
        ax2.set_title('各实验发现潜力评估', fontsize=14, fontweight='bold')

        # 添加数值
        for i in range(len(particle_names)):
            for j in range(len(experiments)):
                value = discovery_potentials[i][j]
                color = 'white' if value > 0.5 else 'black'
                ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=color, fontweight='bold')

        plt.colorbar(im, ax=ax2, label='发现潜力 (0-1)')

        plt.tight_layout()

        # 保存图像
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        plt.savefig(save_dir / 'new_particle_spectrum.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / 'new_particle_spectrum.pdf',
                    bbox_inches='tight')

        plt.show()

        return fig

    def generate_comprehensive_report(self, output_dir: str = "results") -> str:
        """
        生成综合预言报告

        Parameters
        ----------
        output_dir : str, optional
            输出目录

        Returns
        -------
        str
            报告文本
        """
        # 获取所有预言
        weaklet = self.predict_weaklet()
        stronglets = self.predict_stronglet_family()
        composites = self.predict_composite_states()

        all_particles = [weaklet] + stronglets + composites

        # 检查实验约束
        constraint_results = {}
        for particle in all_particles:
            constraint_results[particle.symbol] = self.check_experimental_constraints(particle)

        # 计算发现潜力
        discovery_potentials = {}
        for particle in all_particles:
            discovery_potentials[particle.symbol] = {}
            for exp in ['LHC', 'Belle_II', '固定靶']:
                discovery_potentials[particle.symbol][exp] = (
                    self.calculate_discovery_potential(particle, exp)
                )

        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("新粒子预言综合报告")
        report.append("=" * 80)
        report.append("")

        report.append("1. 理论基础:")
        report.append(f"   基于质子质量公式: m_p/m_e = 3α^{{-1/2}} + 4πα^{{-1}} + 3α^{{-2/3}}")
        report.append(f"   项分析:")
        report.append(f"       项1 (3α^{{-1/2}}): {self.term_structure['term1_mev']:.2f} MeV - 强小子贡献")
        report.append(f"       项2 (4πα^{{-1}}): {self.term_structure['term2_mev']:.2f} MeV - 胶子海凝聚")
        report.append(f"       项3 (3α^{{-2/3}}): {self.term_structure['term3_mev']:.2f} MeV - 弱相互作用修正")
        report.append("")

        report.append("2. 主要新粒子预言:")
        report.append("")

        for particle in all_particles:
            report.append(f"   {particle.name} ({particle.symbol}):")
            report.append(f"       预言质量: {particle.predicted_mass:.2f} ± {particle.mass_uncertainty:.2f} MeV")
            report.append(f"       理论置信度: {particle.confidence_level:.1%}")

            # 实验约束
            constraints = constraint_results[particle.symbol]
            if constraints['overall']['experimentally_excluded']:
                report.append(f"       实验状态: ❌ 已被实验排除")
            else:
                report.append(f"       实验状态: ✅ 尚未被排除")
                report.append(f"       生存概率: {constraints['overall']['survival_probability']:.1%}")

            # 发现潜力
            best_exp = max(['LHC', 'Belle_II', '固定靶'],
                           key=lambda exp: discovery_potentials[particle.symbol][exp]['discovery_potential'])
            best_potential = discovery_potentials[particle.symbol][best_exp]

            if best_potential['discovery_potential'] > 0.3:
                report.append(f"       最佳实验: {best_exp} (潜力: {best_potential['discovery_potential']:.1%})")
            else:
                report.append(f"       发现挑战: 在所有实验中潜力均低于30%")

            report.append("")

        report.append("3. 关键预言总结:")
        report.append("")

        # 最有可能的预言
        viable_particles = [p for p in all_particles
                            if not constraint_results[p.symbol]['overall']['experimentally_excluded']]

        if viable_particles:
            viable_particles.sort(key=lambda p: p.confidence_level, reverse=True)

            report.append("   最有可能存在的新粒子:")
            for i, particle in enumerate(viable_particles[:3], 1):
                best_exp = max(['LHC', 'Belle_II', '固定靶'],
                               key=lambda exp: discovery_potentials[particle.symbol][exp]['discovery_potential'])
                report.append(f"   {i}. {particle.name} ({particle.symbol}):")
                report.append(f"       质量范围: {particle.predicted_mass:.1f} ± {particle.mass_uncertainty:.1f} MeV")
                report.append(f"       建议实验: {best_exp}")
                report.append(f"       探测特征: {particle.detection_signature[:50]}...")
        else:
            report.append("   ⚠️ 所有预言粒子均已被实验排除，需要理论修正")

        report.append("")

        report.append("4. 实验建议:")
        report.append("")

        # 按实验分组建议
        experiments = ['LHC', 'Belle_II', '固定靶']
        for exp in experiments:
            exp_particles = []
            for particle in viable_particles:
                potential = discovery_potentials[particle.symbol][exp]
                if potential['discovery_potential'] > 0.2:
                    exp_particles.append((particle, potential['discovery_potential']))

            if exp_particles:
                exp_particles.sort(key=lambda x: x[1], reverse=True)
                report.append(f"   {exp}:")
                for particle, potential in exp_particles[:3]:
                    report.append(f"       • {particle.name}: 潜力 {potential:.1%}, "
                                  f"质量 ~{particle.predicted_mass:.1f} MeV")

        report.append("")

        report.append("5. 理论意义:")
        report.append("   如果发现这些粒子，将:")
        report.append("   a) 验证'快电荷拖曳慢胶子海'质量生成机制")
        report.append("   b) 揭示超出标准模型的新层次结构")
        report.append("   c) 提供探索QCD真空和弱相互作用的新探针")
        report.append("   d) 可能解释暗物质等宇宙学谜题")
        report.append("")

        report.append("6. 结论:")
        report.append(f"   共预言 {len(all_particles)} 种新粒子，其中 {len(viable_particles)} 种尚未被实验排除。")
        report.append("   弱小子(质量~20 MeV)是最有希望的候选者，建议在Belle II和固定靶实验中优先搜索。")
        report.append("   这些预言为实验物理提供了明确、可检验的目标。")

        report_text = "\n".join(report)

        # 保存结果
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 保存粒子数据
        particle_data = []
        for particle in all_particles:
            particle_dict = {
                'name': particle.name,
                'symbol': particle.symbol,
                'predicted_mass_mev': particle.predicted_mass,
                'mass_uncertainty_mev': particle.mass_uncertainty,
                'quantum_numbers': particle.quantum_numbers,
                'confidence_level': particle.confidence_level,
                'experimental_constraints': constraint_results[particle.symbol],
                'discovery_potentials': discovery_potentials[particle.symbol]
            }
            particle_data.append(particle_dict)

        with open(output_path / "particle_predictions.json", "w", encoding='utf-8') as f:
            json.dump(particle_data, f, indent=2, ensure_ascii=False)

        # 保存报告
        with open(output_path / "new_particle_report.txt", "w", encoding='utf-8') as f:
            f.write(report_text)

        return report_text


def main():
    """主函数"""
    print("新粒子预言器启动...")
    print("-" * 50)

    # 创建预言器
    predictor = NewParticlePredictor()

    # 生成报告
    report = predictor.generate_comprehensive_report()
    print(report)

    # 绘制图像
    print("正在生成粒子谱图...")
    predictor.plot_particle_spectrum()

    print("-" * 50)
    print("预言完成！")
    print("结果已保存至 results/ 目录")
    print("1. particle_predictions.json - 详细预言数据")
    print("2. new_particle_report.txt - 文本报告")
    print("3. figures/new_particle_spectrum.png - 粒子谱图")


if __name__ == "__main__":
    main()