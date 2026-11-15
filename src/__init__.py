"""insitu-tem-dynamics-quant

原位透射电镜（In-situ TEM）纳米颗粒演化序列的仿真生成、亚像素漂移校正与定量动力学分析工具包。

模块划分：
    simulator     —— 物理驱动的原位 TEM 时间序列仿真（Ostwald 粗化 + 热漂移 + 低剂量噪声）
    registration  —— 基于相位相关的亚像素漂移校正
    quantifier    —— 去噪、自适应分割、分水岭分离、逐帧统计与幂律动力学拟合
    visualizer    —— 期刊风格 2×2 综合定量分析图
"""

__version__ = "0.1.0"

__all__ = ["simulator", "registration", "quantifier", "visualizer", "__version__"]
