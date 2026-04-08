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

import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

import matplotlib.pyplot as plt
from io import BytesIO

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        """初始化"""
        self.font_name, self.font_bold_name = self._init_pdf_fonts()
        self.styles = getSampleStyleSheet()
        # 添加自定义样式（如果不存在）
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomTitle', fontName=self.font_bold_name, fontSize=16, alignment=TA_CENTER, spaceAfter=12))
        if 'CustomSubtitle' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomSubtitle', fontName=self.font_bold_name, fontSize=14, alignment=TA_CENTER, spaceAfter=8))
        if 'CustomSection' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomSection', fontName=self.font_bold_name, fontSize=12, alignment=TA_LEFT, spaceAfter=6))
        if 'CustomNormal' not in self.styles:
            self.styles.add(ParagraphStyle(name='CustomNormal', fontName=self.font_name, fontSize=10, alignment=TA_LEFT, spaceAfter=4))
        if 'Right' not in self.styles:
            self.styles.add(ParagraphStyle(name='Right', fontName=self.font_name, fontSize=10, alignment=TA_RIGHT, spaceAfter=4))

    def _init_pdf_fonts(self):
        """
        初始化PDF字体，优先Windows中文字体，失败则回退STSong-Light，再回退Helvetica。
        """
        font_candidates = [
            ("VS_SimHei", os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simhei.ttf")),
            ("VS_MSYH", os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "msyh.ttc")),
            ("VS_SimSun", os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "simsun.ttc")),
        ]

        for alias, path in font_candidates:
            if not os.path.exists(path):
                continue
            try:
                if alias not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(alias, path))
                return alias, alias
            except Exception:
                continue

        try:
            if "STSong-Light" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
            return "STSong-Light", "STSong-Light"
        except Exception:
            return "Helvetica", "Helvetica-Bold"

    @staticmethod
    def _format_assessment_reason(reason_code: str) -> str:
        reason_map = {
            "no_tumor_label": "未检测到肿瘤标签",
            "no_tumor_label_4": "未检测到肿瘤标签4",
            "no_vessel_label": "未检测到血管标签(PV/HV)",
            "exception": "距离计算异常"
        }
        return reason_map.get(reason_code, "原因未知")

    def _two_col_table_style(self, second_col_align: str = 'RIGHT'):
        return TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), second_col_align),
            ('FONTNAME', (0, 0), (-1, -1), self.font_name),
            ('FONTNAME', (0, 0), (0, -1), self.font_bold_name),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
    
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
            table.setStyle(self._two_col_table_style(second_col_align='LEFT'))
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
                table.setStyle(self._two_col_table_style())
                elements.append(table)
                elements.append(Spacer(1, 0.5*cm))
            
            # 血管量化指标
            if 'vascular_metrics' in report_data:
                elements.append(Paragraph('血管量化指标', self.styles['CustomSection']))
                metrics = report_data['vascular_metrics']
                metric_data = []
                metric_names = {
                    'vascular_volume': '血管总体积 (ml)',
                    'portal_volume': '门静脉体积 (ml)',
                    'hepatic_volume': '肝静脉体积 (ml)',
                    'pv_hv_volume_ratio': '门/肝静脉体积比',
                    'vascular_length': '血管总长度 (mm)',
                    'portal_length': '门静脉长度 (mm)',
                    'hepatic_length': '肝静脉长度 (mm)',
                    'mean_diameter': '平均直径 (mm)',
                    'max_diameter': '最大直径 (mm)',
                    'portal_mean_diameter': '门静脉平均直径 (mm)',
                    'hepatic_mean_diameter': '肝静脉平均直径 (mm)',
                    'branch_count': '分支数',
                    'portal_branch_count': '门静脉分支数',
                    'hepatic_branch_count': '肝静脉分支数',
                    'junction_count': '分叉点数量',
                    'vascular_density': '血管密度',
                    'surface_area': '血管表面积 (mm²)',
                    'volume_fraction': '血管体积分数',
                    'mean_branch_angle': '平均分支角度 (°)',
                    'connectivity': '血管连通性',
                    'curvature': '血管弯曲度',
                    'trunk_small_ratio': '主干/细支比例',
                    'micro_branch_voxels': '5级+微小分支体素数',
                }
                
                for key, value in metrics.items():
                    if key in metric_names:
                        if isinstance(value, float):
                            metric_data.append([metric_names[key], f'{value:.2f}'])
                        else:
                            metric_data.append([metric_names[key], str(value)])
                
                if metric_data:
                    table = Table(metric_data, colWidths=[6*cm, 3*cm])
                    table.setStyle(self._two_col_table_style())
                    elements.append(table)
                else:
                    elements.append(Paragraph('无可用血管量化指标', self.styles['CustomNormal']))
                elements.append(Spacer(1, 0.5*cm))
            
            # 手术风险评估
            if 'risk_assessment' in report_data:
                elements.append(Paragraph('手术风险评估', self.styles['CustomSection']))
                risk = report_data['risk_assessment']
                elements.append(Paragraph(f'整体风险: {risk.get("overall_risk", "未知")}', self.styles['CustomNormal']))
                elements.append(Paragraph(f'建议: {risk.get("recommendation", "无")}', self.styles['CustomNormal']))
                elements.append(Spacer(1, 0.5*cm))

            # AI血管结构识别摘要（主干/细支/分叉/5级）
            if 'vessel_analysis' in report_data:
                elements.append(Paragraph('AI血管结构识别摘要', self.styles['CustomSection']))
                va = report_data['vessel_analysis']
                for key, title in [('pv', '门静脉(PV)'), ('hv', '肝静脉(HV)')]:
                    if key in va and isinstance(va[key], dict):
                        item = va[key]
                        levels = item.get('level_counts', {})
                        txt = (
                            f"{title}: 主干={item.get('trunk_voxel_count', 0)}, "
                            f"细支={item.get('small_voxel_count', 0)}, "
                            f"分叉={item.get('junction_voxel_count', 0)}, "
                            f"5级+微小分支={levels.get('level_5_plus', 0)}"
                        )
                        elements.append(Paragraph(txt, self.styles['CustomNormal']))
                elements.append(Spacer(1, 0.5*cm))

            # 切割模拟统计
            if 'cut_stats' in report_data:
                elements.append(Paragraph('手术切割模拟', self.styles['CustomSection']))
                cs = report_data['cut_stats']
                if cs.get('available'):
                    cut_data = [
                        ['剩余肝体积', f"{cs.get('remaining_ml', 0.0):.2f} ml"],
                        ['切除体积', f"{cs.get('resected_ml', 0.0):.2f} ml"],
                        ['剩余比例', f"{cs.get('remaining_ratio', 0.0) * 100:.1f}%"],
                        ['门静脉切断预警', '是' if cs.get('severed_pv') else '否'],
                        ['肝静脉切断预警', '是' if cs.get('severed_hv') else '否'],
                    ]
                else:
                    cut_data = [['状态', '暂无可用切割体积统计']]
                table = Table(cut_data, colWidths=[5*cm, 6*cm])
                table.setStyle(self._two_col_table_style())
                elements.append(table)
                elements.append(Spacer(1, 0.5*cm))
            
            # 添加图像
            if 'images' in report_data:
                elements.append(Paragraph('分割结果可视化', self.styles['CustomSubtitle']))
                elements.append(Spacer(1, 0.5*cm))
                
                for img_name, img_data in report_data['images'].items():
                    elements.append(Paragraph(img_name, self.styles['CustomSection']))
                    # 将numpy数组转换为PIL图像
                    img_buffer = BytesIO()
                    fig = plt.figure(figsize=(6, 4), dpi=150)
                    if isinstance(img_data, np.ndarray) and img_data.ndim == 2:
                        # 标签图（整数）优先彩色显示，普通影像灰度显示
                        if np.issubdtype(img_data.dtype, np.integer) and np.max(img_data) <= 10:
                            plt.imshow(img_data, cmap='jet', vmin=0, vmax=max(1, int(np.max(img_data))))
                        else:
                            plt.imshow(img_data, cmap='gray')
                    else:
                        plt.imshow(img_data)
                    plt.axis('off')
                    plt.savefig(img_buffer, format='png', bbox_inches='tight')
                    plt.close(fig)
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
                'portal_volume': '门静脉体积',
                'hepatic_volume': '肝静脉体积',
                'pv_hv_volume_ratio': '门/肝静脉体积比',
                'vascular_length': '血管总长度',
                'portal_length': '门静脉长度',
                'hepatic_length': '肝静脉长度',
                'mean_diameter': '平均直径',
                'max_diameter': '最大直径',
                'portal_mean_diameter': '门静脉平均直径',
                'hepatic_mean_diameter': '肝静脉平均直径',
                'branch_count': '分支数',
                'portal_branch_count': '门静脉分支数',
                'hepatic_branch_count': '肝静脉分支数',
                'junction_count': '分叉点数量',
                'vascular_density': '血管密度',
                'surface_area': '血管表面积',
                'volume_fraction': '血管体积分数',
                'mean_branch_angle': '平均分支角度',
                'connectivity': '血管连通性',
                'curvature': '血管弯曲度',
                'trunk_small_ratio': '主干/细支比例',
                'micro_branch_voxels': '5级+微小分支体素数',
            }
            for key, value in metrics.items():
                if key in metric_names:
                    if isinstance(value, float):
                        summary.append(f'  {metric_names[key]}: {value:.2f}')
                    else:
                        summary.append(f'  {metric_names[key]}: {value}')
        
        # 手术风险评估
        if 'risk_assessment' in report_data:
            risk = report_data['risk_assessment']
            summary.append('\n手术风险评估:')
            summary.append(f'  整体风险: {risk.get("overall_risk", "未知")}')
            summary.append(f'  建议: {risk.get("recommendation", "无")}')

        # AI结构识别摘要
        if 'vessel_analysis' in report_data:
            va = report_data['vessel_analysis']
            summary.append('\nAI血管结构识别:')
            for key, title in [('pv', '门静脉(PV)'), ('hv', '肝静脉(HV)')]:
                if key in va and isinstance(va[key], dict):
                    item = va[key]
                    levels = item.get('level_counts', {})
                    summary.append(
                        f"  {title}: 主干={item.get('trunk_voxel_count', 0)}, 细支={item.get('small_voxel_count', 0)}, "
                        f"分叉={item.get('junction_voxel_count', 0)}, 5级+={levels.get('level_5_plus', 0)}"
                    )

        # 切割统计
        if 'cut_stats' in report_data:
            cs = report_data['cut_stats']
            summary.append('\n切割模拟:')
            if cs.get('available'):
                summary.append(f"  剩余体积: {cs.get('remaining_ml', 0.0):.2f} ml")
                summary.append(f"  切除体积: {cs.get('resected_ml', 0.0):.2f} ml")
                summary.append(f"  门静脉切断预警: {'是' if cs.get('severed_pv') else '否'}")
                summary.append(f"  肝静脉切断预警: {'是' if cs.get('severed_hv') else '否'}")
            else:
                summary.append("  暂无可用切割体积统计")
        
        # 结论
        summary.append('\n结论:')
        summary.append('本报告基于AI分割结果生成，仅供参考。最终手术决策应结合临床实际情况。')
        
        return '\n'.join(summary)
