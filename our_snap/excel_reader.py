#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Excel表格读取模块
功能：读取522数据库目录下的Excel表格，建立患者ID与表格信息的映射
"""

import os
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path


class ExcelDataReader:
    """读取Excel表格中的患者信息"""
    
    def __init__(self, db_root: str):
        """
        初始化Excel读取器
        
        Args:
            db_root: 数据库根目录（例如 E:\ITK-SNAP 4.2\522）
        """
        self.db_root = db_root
        self.patient_data = {}  # {patient_id: {column_name: value, ...}}
        self.excel_files = []
        self._load_excel_files()
    
    def _load_excel_files(self):
        """扫描数据库根目录下的Excel文件并加载数据"""
        if not os.path.exists(self.db_root):
            return
        
        # 查找所有Excel文件
        excel_extensions = ['.xlsx', '.xls']
        for file in os.listdir(self.db_root):
            # 跳过Excel临时文件（以~$开头的文件）
            if file.startswith('~$'):
                continue
            
            file_path = os.path.join(self.db_root, file)
            if os.path.isfile(file_path):
                ext = Path(file).suffix.lower()
                if ext in excel_extensions:
                    self.excel_files.append(file_path)
                    self._load_excel_file(file_path)
    
    def _load_excel_file(self, file_path: str):
        """加载单个Excel文件"""
        try:
            # 尝试读取所有sheet
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                self._parse_dataframe(df, file_path, sheet_name)
        except Exception as e:
            print(f"读取Excel文件失败 {file_path}: {e}")
    
    def _parse_dataframe(self, df: pd.DataFrame, file_path: str, sheet_name: str):
        """
        解析DataFrame，建立患者ID到数据的映射
        
        假设表格中有一列包含患者ID（可能是"患者ID"、"Patient ID"、"序号"等）
        """
        if df.empty:
            return
        
        # 尝试找到患者ID列（可能的列名）
        patient_id_columns = ['患者ID', 'Patient ID', 'PatientID', '序号', '编号', 'ID', 'id']
        patient_id_col = None
        
        for col in df.columns:
            col_str = str(col).strip()
            if col_str in patient_id_columns:
                patient_id_col = col
                break
        
        # 如果没找到明确的ID列，尝试第一列或包含数字的列
        if patient_id_col is None:
            # 检查第一列是否可能是ID
            first_col = df.columns[0]
            if df[first_col].dtype in ['int64', 'float64']:
                patient_id_col = first_col
            else:
                # 尝试找包含"ID"或"id"的列
                for col in df.columns:
                    if 'id' in str(col).lower() or '编号' in str(col) or '序号' in str(col):
                        patient_id_col = col
                        break
        
        if patient_id_col is None:
            # 如果还是找不到，就假设第一列是ID
            patient_id_col = df.columns[0]
        
        # 将患者ID转换为字符串，并建立映射
        for idx, row in df.iterrows():
            try:
                patient_id = str(row[patient_id_col]).strip()
                if not patient_id or patient_id == 'nan':
                    continue
                
                # 将这一行的所有数据存储
                row_data = {}
                for col in df.columns:
                    value = row[col]
                    # 处理NaN值
                    if pd.isna(value):
                        row_data[str(col)] = ""
                    else:
                        row_data[str(col)] = str(value)
                
                # 添加来源信息
                row_data['_source_file'] = os.path.basename(file_path)
                row_data['_source_sheet'] = sheet_name
                
                # 如果该患者ID已存在，合并数据（后加载的覆盖先加载的）
                if patient_id in self.patient_data:
                    self.patient_data[patient_id].update(row_data)
                else:
                    self.patient_data[patient_id] = row_data
            except Exception as e:
                print(f"解析行数据失败 (行{idx}): {e}")
                continue
    
    def get_patient_info(self, patient_id: str) -> Optional[Dict]:
        """
        获取指定患者ID的表格信息
        
        Args:
            patient_id: 患者ID（字符串）
            
        Returns:
            包含该患者所有表格信息的字典，如果不存在则返回None
        """
        # 尝试直接匹配
        if patient_id in self.patient_data:
            return self.patient_data[patient_id]
        
        # 尝试去掉前导零或后缀匹配
        patient_id_clean = patient_id.lstrip('0')
        if patient_id_clean in self.patient_data:
            return self.patient_data[patient_id_clean]
        
        # 尝试部分匹配（如果patient_id是"86"，表格中可能是"086"）
        for key in self.patient_data.keys():
            if key.lstrip('0') == patient_id or patient_id.lstrip('0') == key:
                return self.patient_data[key]
        
        return None
    
    def get_all_patient_ids(self) -> List[str]:
        """返回所有已加载的患者ID列表"""
        return list(self.patient_data.keys())
    
    def has_data(self) -> bool:
        """检查是否已加载数据"""
        return len(self.patient_data) > 0

