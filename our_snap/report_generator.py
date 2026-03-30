#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
报告生成模块
功能：一键导出包含图文数据与风险评估的PDF手术规划报告
"""

import os
import sys
import datetime
import numpy as np

# 直接导入reportlab库
print("开始导入reportlab库...")
try:
    import reportlab
    print(f"reportlab库导入成功，版本: {reportlab.__version__}")
except ImportError as e:
    print(f"reportlab库导入失败: {e}")
    import traceback
    traceback.print_exc()
    raise

try:
    from reportlab.lib import colors
    print("reportlab.lib.colors导入成功")
except ImportError as e:
    print(f"reportlab.lib.colors导入失败: {e}")
    raise

try:
    from reportlab.lib.pagesizes import A4, landscape
    print("reportlab.lib.pagesizes导入成功")
except ImportError as e:
    print(f"reportlab.lib.pagesizes导入失败: {e}")
    raise

try:
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    print("reportlab.lib.styles导入成功")
except ImportError as e:
    print(f"reportlab.lib.styles导入失败: {e}")
    raise

try:
    from reportlab.lib.units import cm
    print("reportlab.lib.units导入成功")
except ImportError as e:
    print(f"reportlab.lib.units导入失败: {e}")
    raise

try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    print("reportlab.platypus导入成功")
except ImportError as e:
    print(f"reportlab.platypus导入失败: {e}")
    raise

try:
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    print("reportlab.lib.enums导入成功")
except ImportError as e:
    print(f"reportlab.lib.enums导入失败: {e}")
    raise

print("所有reportlab模块导入成功")

import matplotlib.pyplot as plt
from io import BytesIO

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化"""
        self.styles = getSampleStyleSheet()
        # 添加自定义样式（如果不存在）
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomTitle', fontName='Helvetica-Bold', fontSize=16, alignment=TA_CENTER, spaceAfter=12))
        if 'CustomSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomSubtitle', fontName='Helvetica-Bold', fontSize=14, alignment=TA_CENTER, spaceAfter=8))
        if 'CustomSection' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomSection', fontName='Helvetica-Bold', fontSize=12, alignment=TA_LEFT, spaceAfter=6))
        if 'CustomNormal' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomNormal', fontName='Helvetica', fontSize=10, alignment=TA_LEFT, spaceAfter=4))
        if 'Right' not in self.styles:
            self.styles.add(ParagraphStyle(name='Right', fontName='Helvetica', fontSize=10, alignment=TA_RIGHT, spaceAfter=4))
    
    def generate_report(self, report_data: dict, output_path: str) -> bool:
        """
        生成PDF报告
        
        Args:
            report_data: 报告数据字典
            output_path: 输出文件路径
            
        Returns:
            bool: 是否生成成功
        """
        try:
            # 创建PDF文档
            doc = SimpleDocTemplate(output_path, pagesize=landscape(A4))
            elements = []
            
            # 添加标题
            elements.append(Paragraph('肝脏血管分割与手术规划报告', self.styles['CustomTitle']))
            elements.append(Spacer(1, 1*cm))
            
            # 添加基本信息
            elements.append(Paragraph('基本信息', self.styles['CustomSubtitle']))
            elements.append(Spacer(1, 0.5*cm))
            
            # 基本信息表格
            basic_info = [
                ['患者ID', report_data.get('patient_id', 'N/A')],
                ['检查日期', report_data.get('exam_date', datetime.datetime.now().strftime('%Y-%m-%d'))],
                ['报告生成时间', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['分割方法', report_data.get('segmentation_method', 'AI自动分割')],
                ['模型变体', report_data.get('model_variant', 'mobileuvit_l')]
            ]
            
            table = Table(basic_info, colWidths=[4*cm, 10*cm])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 1*cm))
            
            # 添加分割结果
            elements.append(Paragraph('分割结果', self.styles['CustomSubtitle']))
            elements.append(Spacer(1, 0.5*cm))
            
            # 分割统计
            if 'volumes' in report_data:
                elements.append(Paragraph('肝脏体积分析', self.styles['CustomSection']))
                volumes = report_data['volumes']
                volume_data = []
                for key, value in volumes.items():
                    if key != 'total_liver':
                        volume_data.append([key.replace('segment_', '段 '), f'{value:.2f} ml'])
                volume_data.append(['总体积', f'{volumes.get("total_liver", 0):.2f} ml'])
                
                table = Table(volume_data, colWidths=[4*cm, 4*cm])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.5*cm))
            
            # 血管量化指标
            if 'vascular_metrics' in report_data:
                elements.append(Paragraph('血管量化指标', self.styles['CustomSection']))
                metrics = report_data['vascular_metrics']
                metric_data = []
                metric_names = {
                    'vascular_volume': '血管总体积 (ml)',
                    'vascular_length': '血管总长度 (mm)',
                    'mean_diameter': '平均直径 (mm)',
                    'max_diameter': '最大直径 (mm)',
                    'branch_count': '分支数',
                    'vascular_density': '血管密度',
                    'surface_area': '血管表面积 (mm²)',
                    'volume_fraction': '血管体积分数',
                    'mean_branch_angle': '平均分支角度 (°)',
                    'connectivity': '血管连通性',
                    'curvature': '血管弯曲度'
                }
                
                for key, value in metrics.items():
                    if key in metric_names:
                        if isinstance(value, float):
                            metric_data.append([metric_names[key], f'{value:.2f}'])
                        else:
                            metric_data.append([metric_names[key], str(value)])
                
                table = Table(metric_data, colWidths=[6*cm, 3*cm])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.5*cm))
            
            # 肿瘤-血管距离评估
            if 'tumor_vessel_assessment' in report_data:
                elements.append(Paragraph('肿瘤-血管距离评估', self.styles['CustomSection']))
                assessment = report_data['tumor_vessel_assessment']
                risk_data = [
                    ['最小距离', f'{assessment.get("min_distance", 0):.2f} mm'],
                    ['风险等级', assessment.get("risk_level", '未知')],
                    ['风险评分', str(assessment.get("risk_score", 0))]
                ]
                
                table = Table(risk_data, colWidths=[4*cm, 4*cm])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.5*cm))
            
            # 手术风险评估
            if 'risk_assessment' in report_data:
                elements.append(Paragraph('手术风险评估', self.styles['CustomSection']))
                risk = report_data['risk_assessment']
                elements.append(Paragraph(f'整体风险: {risk.get("overall_risk", "未知")}', self.styles['CustomNormal']))
                elements.append(Paragraph(f'建议: {risk.get("recommendation", "无")}', self.styles['CustomNormal']))
                elements.append(Spacer(1, 0.5*cm))
            
            # 添加图像
            if 'images' in report_data:
                elements.append(Paragraph('分割结果可视化', self.styles['CustomSubtitle']))
                elements.append(Spacer(1, 0.5*cm))
                
                for img_name, img_data in report_data['images'].items():
                    elements.append(Paragraph(img_name, self.styles['CustomSection']))
                    # 将numpy数组转换为PIL图像
                    img_buffer = BytesIO()
                    plt.imshow(img_data, cmap='gray')
                    plt.axis('off')
                    plt.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0)
                    
                    # 添加图像到报告
                    img = Image(img_buffer, width=15*cm, height=10*cm)
                    elements.append(img)
                    elements.append(Spacer(1, 0.5*cm))
            
            # 添加结论
            elements.append(Paragraph('结论', self.styles['CustomSubtitle']))
            elements.append(Spacer(1, 0.5*cm))
            elements.append(Paragraph('本报告基于AI分割结果生成，仅供参考。最终手术决策应结合临床实际情况。', self.styles['CustomNormal']))
            
            # 生成PDF
            doc.build(elements)
            print(f"报告生成成功: {output_path}")
            return True
        except Exception as e:
            print(f"报告生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_summary_report(self, report_data: dict) -> str:
        """
        生成简要报告（文本格式）
        
        Args:
            report_data: 报告数据
            
        Returns:
            简要报告文本
        """
        summary = []
        summary.append('肝脏血管分割与手术规划报告')
        summary.append('=' * 50)
        
        # 基本信息
        summary.append('\n基本信息:')
        summary.append(f'患者ID: {report_data.get("patient_id", "N/A")}')
        summary.append(f'检查日期: {report_data.get("exam_date", datetime.datetime.now().strftime("%Y-%m-%d"))}')
        summary.append(f'报告生成时间: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        summary.append(f'分割方法: {report_data.get("segmentation_method", "AI自动分割")}')
        summary.append(f'模型变体: {report_data.get("model_variant", "mobileuvit_l")}')
        
        # 分割结果
        summary.append('\n分割结果:')
        if 'volumes' in report_data:
            volumes = report_data['volumes']
            summary.append('肝脏体积分析:')
            for key, value in volumes.items():
                if key != 'total_liver':
                    summary.append(f'  {key.replace("segment_", "段 ")}: {value:.2f} ml')
            summary.append(f'  总体积: {volumes.get("total_liver", 0):.2f} ml')
        
        # 血管量化指标
        if 'vascular_metrics' in report_data:
            metrics = report_data['vascular_metrics']
            summary.append('\n血管量化指标:')
            metric_names = {
                'vascular_volume': '血管总体积',
                'vascular_length': '血管总长度',
                'mean_diameter': '平均直径',
                'max_diameter': '最大直径',
                'branch_count': '分支数',
                'vascular_density': '血管密度',
                'surface_area': '血管表面积',
                'volume_fraction': '血管体积分数',
                'mean_branch_angle': '平均分支角度',
                'connectivity': '血管连通性',
                'curvature': '血管弯曲度'
            }
            for key, value in metrics.items():
                if key in metric_names:
                    if isinstance(value, float):
                        summary.append(f'  {metric_names[key]}: {value:.2f}')
                    else:
                        summary.append(f'  {metric_names[key]}: {value}')
        
        # 肿瘤-血管距离评估
        if 'tumor_vessel_assessment' in report_data:
            assessment = report_data['tumor_vessel_assessment']
            summary.append('\n肿瘤-血管距离评估:')
            summary.append(f'  最小距离: {assessment.get("min_distance", 0):.2f} mm')
            summary.append(f'  风险等级: {assessment.get("risk_level", "未知")}')
            summary.append(f'  风险评分: {assessment.get("risk_score", 0)}')
        
        # 手术风险评估
        if 'risk_assessment' in report_data:
            risk = report_data['risk_assessment']
            summary.append('\n手术风险评估:')
            summary.append(f'  整体风险: {risk.get("overall_risk", "未知")}')
            summary.append(f'  建议: {risk.get("recommendation", "无")}')
        
        # 结论
        summary.append('\n结论:')
        summary.append('本报告基于AI分割结果生成，仅供参考。最终手术决策应结合临床实际情况。')
        
        return '\n'.join(summary)