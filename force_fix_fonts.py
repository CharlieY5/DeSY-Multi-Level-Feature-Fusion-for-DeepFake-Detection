#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强制修复matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
import shutil

def force_setup_chinese_fonts():
    """强制设置中文字体"""
    system = platform.system()
    
    # 清除matplotlib字体缓存
    import matplotlib
    fm._rebuild()
    
    if system == "Windows":
        # Windows系统字体路径
        font_paths = [
            r'C:\Windows\Fonts\simhei.ttf',      # 黑体
            r'C:\Windows\Fonts\msyh.ttc',        # 微软雅黑
            r'C:\Windows\Fonts\simsun.ttc',      # 宋体
            r'C:\Windows\Fonts\simkai.ttf',      # 楷体
            r'C:\Windows\Fonts\simfang.ttf',     # 仿宋
        ]
        
        # 检查字体文件是否存在
        available_fonts = []
        for font_path in font_paths:
            if os.path.exists(font_path):
                available_fonts.append(font_path)
                print(f"找到字体文件: {font_path}")
            else:
                print(f"字体文件不存在: {font_path}")
        
        if available_fonts:
            # 强制添加字体
            for font_path in available_fonts:
                try:
                    fm.fontManager.addfont(font_path)
                    print(f"成功添加字体: {font_path}")
                except Exception as e:
                    print(f"添加字体失败: {font_path}, 错误: {e}")
            
            # 设置字体优先级
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 重新构建字体管理器
            fm._rebuild()
            
            print("字体配置完成")
            return True
        else:
            print("未找到任何中文字体文件")
            return False
    
    else:
        # 其他系统使用默认配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return False

def test_chinese_display():
    """测试中文显示"""
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 测试文本
    test_texts = [
        "中文标题测试 - Chinese Title Test",
        "模型性能对比 - Model Performance Comparison", 
        "准确率: 0.85 - Accuracy: 0.85",
        "融合权重分析 - Fusion Weight Analysis",
        "数据集大小影响 - Dataset Size Impact"
    ]
    
    # 绘制测试文本
    for i, text in enumerate(test_texts):
        ax.text(0.1, 0.8 - i*0.15, text, fontsize=14, transform=ax.transAxes)
    
    # 设置标题和标签
    ax.set_title("中文字体测试 - Chinese Font Test", fontsize=16, fontweight='bold')
    ax.set_xlabel("横轴标签 - X Axis Label", fontsize=12)
    ax.set_ylabel("纵轴标签 - Y Axis Label", fontsize=12)
    
    # 添加图例
    ax.plot([0.2, 0.8], [0.3, 0.7], 'o-', label='测试数据 - Test Data', linewidth=2)
    ax.legend(fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # 保存测试图片
    os.makedirs('charts', exist_ok=True)
    plt.savefig('charts/chinese_font_test_force.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("强制字体测试完成，请检查 charts/chinese_font_test_force.png")

def create_simple_test():
    """创建简单的测试图表"""
    # 强制设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建简单图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 数据
    models = ['Low-Level', 'Mid-Level', 'High-Level', 'Fusion']
    accuracies = [0.85, 0.78, 0.82, 0.91]
    
    # 绘制柱状图
    bars = ax.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    
    # 设置标题和标签
    ax.set_title('模型准确率对比 - Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('模型类型 - Model Type', fontsize=12)
    ax.set_ylabel('准确率 - Accuracy', fontsize=12)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('charts/simple_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("简单测试图表已保存为: charts/simple_test.png")

if __name__ == "__main__":
    print("开始强制修复中文字体...")
    
    # 强制设置字体
    success = force_setup_chinese_fonts()
    
    if success:
        print("字体设置成功，开始测试...")
        # 测试中文显示
        test_chinese_display()
        # 创建简单测试
        create_simple_test()
    else:
        print("字体设置失败，尝试使用默认配置...")
        # 即使失败也尝试创建测试
        create_simple_test()
    
    print("修复完成！") 