"""
主程序：运行所有计算
"""
from proton_mass_formula import verify_formula
from lepton_mass_fit import fit_lepton_masses


def main():
    print("开始执行质量-常数关系分析")
    print("=" * 60)

    # 1. 验证质子质量公式
    verify_formula()

    # 2. 拟合轻子质量
    fit_lepton_masses()

    print("\n" + "=" * 60)
    print("所有计算完成！")


if __name__ == "__main__":
    main()