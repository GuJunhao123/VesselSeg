#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
影像读写模块
功能：支持NIfTI、DICOM、NRRD格式的读取和保存，保留空间参数
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict, List

class ImageIO:
    """影像读写类 - 完全自研实现"""
    
    def __init__(self):
        """初始化"""
        self.current_image_data = None  # numpy数组
        self.current_image_affine = None  # 仿射变换矩阵
        self.current_image_spacing = None  # 像素间距
        self.current_image_origin = None  # 原点
        self.current_image_direction = None  # 方向矩阵
        self.current_dicom_info = None  # 当前图像的DICOM元信息（如果有）
        
    def load_image(self, filepath: str) -> Tuple[Optional[np.ndarray], Dict]:
        """
        加载医学影像文件
        
        Args:
            filepath: 文件路径
            
        Returns:
            (image_data, metadata): 图像数据和元数据字典
                image_data: numpy数组 (H, W, D)
                metadata: 包含affine, spacing, origin, direction等信息
        """
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()
        
        try:
            # 1) 先处理目录：作为DICOM序列读取
            if os.path.isdir(filepath):
                # DICOM序列（文件夹）
                return self._load_dicom(filepath)
            
            # 2) 再根据扩展名分支
            if ext in ['.nii', '.gz']:
                # NIfTI格式
                return self._load_nifti(filepath)
            elif ext in ['.nrrd', '.nhdr']:
                # NRRD格式
                return self._load_nrrd(filepath)
            elif ext in ['.dcm', '.dicom'] or ext == '':
                # DICOM格式：
                # - 有.dcm/.dicom后缀的单个文件
                # - **无扩展名但实际为DICOM的文件（如IM0, IM1等）**
                #
                # 为了获得整套体积，而不是单张切片，
                # 这里使用「文件所在的文件夹」作为DICOM序列路径
                dicom_dir = os.path.dirname(filepath) or '.'
                return self._load_dicom(dicom_dir)
            else:
                # 其他未知格式，最后再尝试一次按DICOM读取
                try:
                    return self._load_dicom(filepath)
                except Exception:
                    raise ValueError(f"不支持的文件格式: {ext or '无扩展名'}")
                
        except Exception as e:
            print(f"加载影像失败: {e}")
            return None, {}
    
    def _load_nifti(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """加载NIfTI文件"""
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()
        
        # 获取空间参数
        affine = nii_img.affine
        header = nii_img.header
        
        # 提取spacing和origin
        spacing = header.get_zooms()[:3]  # 前3个维度
        origin = affine[:3, 3]
        direction = affine[:3, :3]
        
        # 归一化方向矩阵
        for i in range(3):
            norm = np.linalg.norm(direction[:, i])
            if norm > 0:
                direction[:, i] /= norm
        
        metadata = {
            'affine': affine,
            'spacing': spacing,
            'origin': origin,
            'direction': direction,
            'header': header
        }
        
        self.current_image_data = data
        self.current_image_affine = affine
        self.current_image_spacing = spacing
        self.current_image_origin = origin
        self.current_image_direction = direction
        
        return data, metadata
    
    def _load_nrrd(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """加载NRRD文件"""
        sitk_image = sitk.ReadImage(filepath)
        
        # 转换为numpy数组
        data = sitk.GetArrayFromImage(sitk_image)
        # SimpleITK返回的是(D, H, W)，需要转换为(H, W, D)
        data = np.transpose(data, (1, 2, 0))
        
        # 获取空间参数
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        # 构建仿射变换矩阵
        direction_matrix = np.array(direction).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction_matrix * np.array(spacing)
        affine[:3, 3] = origin
        
        metadata = {
            'affine': affine,
            'spacing': spacing,
            'origin': origin,
            'direction': direction_matrix,
            'sitk_image': sitk_image
        }
        
        self.current_image_data = data
        self.current_image_affine = affine
        self.current_image_spacing = spacing
        self.current_image_origin = origin
        self.current_image_direction = direction_matrix
        
        return data, metadata
    
    def _load_dicom(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """加载DICOM文件或序列（目录或单文件）"""
        first_dicom_file = None
        if os.path.isdir(filepath):
            # DICOM序列（可能包含多个SeriesUID）。若 GDCM 找不到，则尝试手动按 IM* 兜底。
            reader = sitk.ImageSeriesReader()
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(filepath)
            except Exception:
                dicom_names = []
            if not dicom_names:
                # 手动收集 IM* 文件
                names = sorted(os.listdir(filepath))
                dicom_names = [
                    os.path.join(filepath, n)
                    for n in names
                    if os.path.isfile(os.path.join(filepath, n)) and n.startswith("IM")
                ]
            if not dicom_names:
                raise ValueError("未找到可加载的DICOM切片")
            first_dicom_file = dicom_names[0]  # 保存第一个文件路径，用于读取完整元数据
            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()
        else:
            # 单个DICOM文件
            first_dicom_file = filepath
            sitk_image = sitk.ReadImage(filepath)
        
        # 转换为numpy数组
        data = sitk.GetArrayFromImage(sitk_image)
        # 转换为(H, W, D)
        data = np.transpose(data, (1, 2, 0))
        
        # 获取空间参数
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        # 构建仿射变换矩阵
        direction_matrix = np.array(direction).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction_matrix * np.array(spacing)
        affine[:3, 3] = origin
        
        # 提取DICOM元信息：优先从第一个文件读取完整元数据（与数据库索引方式一致）
        dicom_info = {}
        if first_dicom_file:
            try:
                # 使用ImageFileReader读取完整元数据（与数据库索引方式一致）
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(first_dicom_file)
                file_reader.LoadPrivateTagsOn()
                file_reader.ReadImageInformation()
                
                def get_tag(tag: str) -> Optional[str]:
                    try:
                        return file_reader.GetMetaData(tag) if file_reader.HasMetaDataKey(tag) else None
                    except Exception:
                        return None
                
                # 提取所有DICOM标签（与数据库索引方式一致）
                dicom_info["PatientID"] = get_tag("0010|0020")
                dicom_info["PatientName"] = get_tag("0010|0010")
                dicom_info["PatientSex"] = get_tag("0010|0040")
                dicom_info["PatientBirthDate"] = get_tag("0010|0030")
                dicom_info["StudyInstanceUID"] = get_tag("0020|000d")
                dicom_info["StudyID"] = get_tag("0020|0010")
                dicom_info["StudyDate"] = get_tag("0008|0020")
                dicom_info["SeriesInstanceUID"] = get_tag("0020|000e")
                dicom_info["SeriesNumber"] = get_tag("0020|0011")
                dicom_info["SeriesDescription"] = get_tag("0008|103e")
                dicom_info["Modality"] = get_tag("0008|0060")
                dicom_info["BodyPartExamined"] = get_tag("0018|0015")
                
                # 去掉值为None的键
                dicom_info = {k: v for k, v in dicom_info.items() if v is not None}
            except Exception:
                # 如果从文件读取失败，回退到从图像对象提取
                dicom_info = self._extract_dicom_info(sitk_image)
        else:
            # 如果没有文件路径，从图像对象提取
            dicom_info = self._extract_dicom_info(sitk_image)

        metadata = {
            'affine': affine,
            'spacing': spacing,
            'origin': origin,
            'direction': direction_matrix,
            'sitk_image': sitk_image,
            'dicom_info': dicom_info
        }
        
        self.current_image_data = data
        self.current_image_affine = affine
        self.current_image_spacing = spacing
        self.current_image_origin = origin
        self.current_image_direction = direction_matrix
        self.current_dicom_info = dicom_info
        
        return data, metadata

    def load_dicom_series(self, file_list: List[str]) -> Tuple[Optional[np.ndarray], Dict]:
        """
        按指定文件列表加载一个 DICOM Series（用于精确到 SeriesUID 的加载）
        """
        if not file_list:
            return None, {}
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_list)
            sitk_image = reader.Execute()

            # 转换为numpy数组
            data = sitk.GetArrayFromImage(sitk_image)
            data = np.transpose(data, (1, 2, 0))

            # 获取空间参数
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()

            direction_matrix = np.array(direction).reshape(3, 3)
            affine = np.eye(4)
            affine[:3, :3] = direction_matrix * np.array(spacing)
            affine[:3, 3] = origin

            # 提取部分DICOM元信息
            dicom_info = self._extract_dicom_info(sitk_image)

            metadata = {
                'affine': affine,
                'spacing': spacing,
                'origin': origin,
                'direction': direction_matrix,
                'sitk_image': sitk_image,
                'dicom_info': dicom_info
            }

            self.current_image_data = data
            self.current_image_affine = affine
            self.current_image_spacing = spacing
            self.current_image_origin = origin
            self.current_image_direction = direction_matrix
            self.current_dicom_info = dicom_info

            return data, metadata
        except Exception as e:
            print(f"加载影像失败: {e}")
            return None, {}

    def _extract_dicom_info(self, sitk_image: sitk.Image) -> Dict:
        """
        从SimpleITK图像中提取常用DICOM信息（若非DICOM则返回空字典）
        """
        info: Dict = {}
        try:
            keys = list(sitk_image.GetMetaDataKeys())
        except Exception:
            return info

        def get(tag: str) -> Optional[str]:
            try:
                return sitk_image.GetMetaData(tag) if tag in keys else None
            except Exception:
                return None

        info["PatientID"] = get("0010|0020")
        info["PatientName"] = get("0010|0010")
        info["PatientSex"] = get("0010|0040")
        info["PatientBirthDate"] = get("0010|0030")

        info["StudyInstanceUID"] = get("0020|000d")
        info["StudyID"] = get("0020|0010")
        info["StudyDate"] = get("0008|0020")

        info["SeriesInstanceUID"] = get("0020|000e")
        info["SeriesNumber"] = get("0020|0011")
        info["SeriesDescription"] = get("0008|103e")

        info["Modality"] = get("0008|0060")
        info["BodyPartExamined"] = get("0018|0015")

        # 去掉值为None的键
        return {k: v for k, v in info.items() if v is not None}
    
    def save_image(self, data: np.ndarray, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        保存影像文件
        
        Args:
            data: 图像数据 (H, W, D)
            filepath: 保存路径
            metadata: 元数据（包含affine, spacing等）
            
        Returns:
            成功返回True，失败返回False
        """
        try:
            ext = Path(filepath).suffix.lower()
            
            if ext in ['.nii', '.gz']:
                return self._save_nifti(data, filepath, metadata)
            elif ext in ['.nrrd']:
                return self._save_nrrd(data, filepath, metadata)
            else:
                # 默认保存为NIfTI
                return self._save_nifti(data, filepath, metadata)
                
        except Exception as e:
            print(f"保存影像失败: {e}")
            return False
    
    def _save_nifti(self, data: np.ndarray, filepath: str, metadata: Optional[Dict]) -> bool:
        """保存NIfTI文件"""
        # 使用metadata中的affine，如果没有则使用默认值
        if metadata and 'affine' in metadata:
            affine = metadata['affine']
        elif self.current_image_affine is not None:
            affine = self.current_image_affine
        else:
            # 默认仿射矩阵
            affine = np.eye(4)
        
        # 创建NIfTI图像对象
        nii_img = nib.Nifti1Image(data, affine)
        
        # 保存（自动处理.gz压缩）
        nib.save(nii_img, filepath)
        return True
    
    def _save_nrrd(self, data: np.ndarray, filepath: str, metadata: Optional[Dict]) -> bool:
        """保存NRRD文件"""
        # 转换为SimpleITK格式 (D, H, W)
        data_transposed = np.transpose(data, (2, 0, 1))
        sitk_image = sitk.GetImageFromArray(data_transposed)
        
        # 设置空间参数
        if metadata:
            if 'spacing' in metadata:
                sitk_image.SetSpacing(metadata['spacing'])
            if 'origin' in metadata:
                sitk_image.SetOrigin(metadata['origin'])
            if 'direction' in metadata:
                direction = metadata['direction'].flatten()
                sitk_image.SetDirection(direction)
        elif self.current_image_spacing is not None:
            sitk_image.SetSpacing(self.current_image_spacing)
            if self.current_image_origin is not None:
                sitk_image.SetOrigin(self.current_image_origin)
            if self.current_image_direction is not None:
                sitk_image.SetDirection(self.current_image_direction.flatten())
        
        # 保存
        sitk.WriteImage(sitk_image, filepath)
        return True
    
    def load_label_map(self, filepath: str) -> Optional[np.ndarray]:
        """
        加载标签掩码文件
        
        Args:
            filepath: 标签文件路径
            
        Returns:
            标签数据 (H, W, D)，数据类型为uint8
        """
        data, _ = self.load_image(filepath)
        if data is not None:
            # 确保为uint8类型
            data = data.astype(np.uint8)
        return data
    
    def save_label_map(self, label_data: np.ndarray, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        保存标签掩码文件
        
        Args:
            label_data: 标签数据 (H, W, D)，应为uint8类型
            filepath: 保存路径
            metadata: 元数据（应与原始影像一致）
            
        Returns:
            成功返回True，失败返回False
        """
        # 确保为uint8类型
        if label_data.dtype != np.uint8:
            label_data = label_data.astype(np.uint8)
        
        return self.save_image(label_data, filepath, metadata)

