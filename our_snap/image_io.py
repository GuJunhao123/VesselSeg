#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
褰卞儚璇诲啓妯″潡
鍔熻兘锛氭敮鎸丯IfTI銆丏ICOM銆丯RRD鏍煎紡鐨勮鍙栧拰淇濆瓨锛屼繚鐣欑┖闂村弬鏁?
"""

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional, Dict, List

try:
    from nibabel.processing import resample_from_to
except Exception:
    resample_from_to = None

class ImageIO:
    """褰卞儚璇诲啓绫?- 瀹屽叏鑷爺瀹炵幇"""
    
    def __init__(self):
        """初始化"""
        self.current_image_data = None  # numpy鏁扮粍
        self.current_image_affine = None  # 浠垮皠鍙樻崲鐭╅樀
        self.current_image_spacing = None  # 鍍忕礌闂磋窛
        self.current_image_origin = None  # 鍘熺偣
        self.current_image_direction = None  # 鏂瑰悜鐭╅樀
        self.current_dicom_info = None  # 褰撳墠鍥惧儚鐨凞ICOM鍏冧俊鎭紙濡傛灉鏈夛級
        
    def load_image(self, filepath: str) -> Tuple[Optional[np.ndarray], Dict]:
        """
        鍔犺浇鍖诲褰卞儚鏂囦欢
        
        Args:
            filepath: 鏂囦欢璺緞
            
        Returns:
            (image_data, metadata): 鍥惧儚鏁版嵁鍜屽厓鏁版嵁瀛楀吀
                image_data: numpy鏁扮粍 (H, W, D)
                metadata: 鍖呭惈affine, spacing, origin, direction绛変俊鎭?
        """
        filepath = str(filepath)
        ext = Path(filepath).suffix.lower()
        
        try:
            # 1) 鍏堝鐞嗙洰褰曪細浣滀负DICOM搴忓垪璇诲彇
            if os.path.isdir(filepath):
                # DICOM搴忓垪锛堟枃浠跺す锛?
                return self._load_dicom(filepath)
            
            # 2) 鍐嶆牴鎹墿灞曞悕鍒嗘敮
            if ext in ['.nii', '.gz']:
                # NIfTI鏍煎紡
                return self._load_nifti(filepath)
            elif ext in ['.nrrd', '.nhdr']:
                # NRRD鏍煎紡
                return self._load_nrrd(filepath)
            elif ext in ['.dcm', '.dicom'] or ext == '':
                # DICOM鏍煎紡锛?
                # - 鏈?dcm/.dicom鍚庣紑鐨勫崟涓枃浠?
                # - **鏃犳墿灞曞悕浣嗗疄闄呬负DICOM鐨勬枃浠讹紙濡侷M0, IM1绛夛級**
                #
                # 涓轰簡鑾峰緱鏁村浣撶Н锛岃€屼笉鏄崟寮犲垏鐗囷紝
                # 杩欓噷浣跨敤銆屾枃浠舵墍鍦ㄧ殑鏂囦欢澶广€嶄綔涓篋ICOM搴忓垪璺緞
                dicom_dir = os.path.dirname(filepath) or '.'
                return self._load_dicom(dicom_dir)
            else:
                # 鍏朵粬鏈煡鏍煎紡锛屾渶鍚庡啀灏濊瘯涓€娆℃寜DICOM璇诲彇
                try:
                    return self._load_dicom(filepath)
                except Exception:
                    raise ValueError(f"涓嶆敮鎸佺殑鏂囦欢鏍煎紡: {ext or '鏃犳墿灞曞悕'}")
                
        except Exception as e:
            print(f"鍔犺浇褰卞儚澶辫触: {e}")
            return None, {}
    
    def _load_nifti(self, filepath: str) -> Tuple[np.ndarray, Dict]:
        """鍔犺浇NIfTI鏂囦欢"""
        nii_img = nib.load(filepath)
        data = nii_img.get_fdata()
        
        # 鑾峰彇绌洪棿鍙傛暟
        affine = nii_img.affine
        header = nii_img.header
        
        # 鎻愬彇spacing鍜宱rigin
        spacing = header.get_zooms()[:3]  # 鍓?涓淮搴?
        origin = affine[:3, 3]
        direction = affine[:3, :3]
        
        # 褰掍竴鍖栨柟鍚戠煩闃?
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
        """鍔犺浇NRRD鏂囦欢"""
        sitk_image = sitk.ReadImage(filepath)
        
        # 杞崲涓簄umpy鏁扮粍
        data = sitk.GetArrayFromImage(sitk_image)
        # SimpleITK杩斿洖鐨勬槸(D, H, W)锛岄渶瑕佽浆鎹负(H, W, D)
        data = np.transpose(data, (1, 2, 0))
        
        # 鑾峰彇绌洪棿鍙傛暟
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        # 鏋勫缓浠垮皠鍙樻崲鐭╅樀
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
        """加载 DICOM 文件或序列（目录或单文件）。"""
        first_dicom_file = None
        if os.path.isdir(filepath):
            # DICOM搴忓垪锛堝彲鑳藉寘鍚涓猄eriesUID锛夈€傝嫢 GDCM 鎵句笉鍒帮紝鍒欏皾璇曟墜鍔ㄦ寜 IM* 鍏滃簳銆?
            reader = sitk.ImageSeriesReader()
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(filepath)
            except Exception:
                dicom_names = []
            if not dicom_names:
                # 鎵嬪姩鏀堕泦 IM* 鏂囦欢
                names = sorted(os.listdir(filepath))
                dicom_names = [
                    os.path.join(filepath, n)
                    for n in names
                    if os.path.isfile(os.path.join(filepath, n)) and n.startswith("IM")
                ]
            if not dicom_names:
                raise ValueError("鏈壘鍒板彲鍔犺浇鐨凞ICOM鍒囩墖")
            first_dicom_file = dicom_names[0]  # 淇濆瓨绗竴涓枃浠惰矾寰勶紝鐢ㄤ簬璇诲彇瀹屾暣鍏冩暟鎹?
            reader.SetFileNames(dicom_names)
            sitk_image = reader.Execute()
        else:
            # 鍗曚釜DICOM鏂囦欢
            first_dicom_file = filepath
            sitk_image = sitk.ReadImage(filepath)
        
        # 杞崲涓簄umpy鏁扮粍
        data = sitk.GetArrayFromImage(sitk_image)
        # 杞崲涓?H, W, D)
        data = np.transpose(data, (1, 2, 0))
        
        # 鑾峰彇绌洪棿鍙傛暟
        spacing = sitk_image.GetSpacing()
        origin = sitk_image.GetOrigin()
        direction = sitk_image.GetDirection()
        
        # 鏋勫缓浠垮皠鍙樻崲鐭╅樀
        direction_matrix = np.array(direction).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction_matrix * np.array(spacing)
        affine[:3, 3] = origin
        
        # 鎻愬彇DICOM鍏冧俊鎭細浼樺厛浠庣涓€涓枃浠惰鍙栧畬鏁村厓鏁版嵁锛堜笌鏁版嵁搴撶储寮曟柟寮忎竴鑷达級
        dicom_info = {}
        if first_dicom_file:
            try:
                # 浣跨敤ImageFileReader璇诲彇瀹屾暣鍏冩暟鎹紙涓庢暟鎹簱绱㈠紩鏂瑰紡涓€鑷达級
                file_reader = sitk.ImageFileReader()
                file_reader.SetFileName(first_dicom_file)
                file_reader.LoadPrivateTagsOn()
                file_reader.ReadImageInformation()
                
                def get_tag(tag: str) -> Optional[str]:
                    try:
                        return file_reader.GetMetaData(tag) if file_reader.HasMetaDataKey(tag) else None
                    except Exception:
                        return None
                
                # 鎻愬彇鎵€鏈塂ICOM鏍囩锛堜笌鏁版嵁搴撶储寮曟柟寮忎竴鑷达級
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
                
                # 鍘绘帀鍊间负None鐨勯敭
                dicom_info = {k: v for k, v in dicom_info.items() if v is not None}
            except Exception:
                # 濡傛灉浠庢枃浠惰鍙栧け璐ワ紝鍥為€€鍒颁粠鍥惧儚瀵硅薄鎻愬彇
                dicom_info = self._extract_dicom_info(sitk_image)
        else:
            # 濡傛灉娌℃湁鏂囦欢璺緞锛屼粠鍥惧儚瀵硅薄鎻愬彇
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
        鎸夋寚瀹氭枃浠跺垪琛ㄥ姞杞戒竴涓?DICOM Series锛堢敤浜庣簿纭埌 SeriesUID 鐨勫姞杞斤級
        """
        if not file_list:
            return None, {}
        try:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(file_list)
            sitk_image = reader.Execute()

            # 杞崲涓簄umpy鏁扮粍
            data = sitk.GetArrayFromImage(sitk_image)
            data = np.transpose(data, (1, 2, 0))

            # 鑾峰彇绌洪棿鍙傛暟
            spacing = sitk_image.GetSpacing()
            origin = sitk_image.GetOrigin()
            direction = sitk_image.GetDirection()

            direction_matrix = np.array(direction).reshape(3, 3)
            affine = np.eye(4)
            affine[:3, :3] = direction_matrix * np.array(spacing)
            affine[:3, 3] = origin

            # 鎻愬彇閮ㄥ垎DICOM鍏冧俊鎭?
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
            print(f"鍔犺浇褰卞儚澶辫触: {e}")
            return None, {}

    def _extract_dicom_info(self, sitk_image: sitk.Image) -> Dict:
        """
        浠嶴impleITK鍥惧儚涓彁鍙栧父鐢―ICOM淇℃伅锛堣嫢闈濪ICOM鍒欒繑鍥炵┖瀛楀吀锛?
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

        # 鍘绘帀鍊间负None鐨勯敭
        return {k: v for k, v in info.items() if v is not None}
    
    def save_image(self, data: np.ndarray, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        淇濆瓨褰卞儚鏂囦欢
        
        Args:
            data: 鍥惧儚鏁版嵁 (H, W, D)
            filepath: 淇濆瓨璺緞
            metadata: 鍏冩暟鎹紙鍖呭惈affine, spacing绛夛級
            
        Returns:
            鎴愬姛杩斿洖True锛屽け璐ヨ繑鍥濬alse
        """
        try:
            ext = Path(filepath).suffix.lower()
            
            if ext in ['.nii', '.gz']:
                return self._save_nifti(data, filepath, metadata)
            elif ext in ['.nrrd']:
                return self._save_nrrd(data, filepath, metadata)
            else:
                # 榛樿淇濆瓨涓篘IfTI
                return self._save_nifti(data, filepath, metadata)
                
        except Exception as e:
            print(f"淇濆瓨褰卞儚澶辫触: {e}")
            return False
    
    def _save_nifti(self, data: np.ndarray, filepath: str, metadata: Optional[Dict]) -> bool:
        """淇濆瓨NIfTI鏂囦欢"""
        # 浣跨敤metadata涓殑affine锛屽鏋滄病鏈夊垯浣跨敤榛樿鍊?
        if metadata and 'affine' in metadata:
            affine = metadata['affine']
        elif self.current_image_affine is not None:
            affine = self.current_image_affine
        else:
            # 榛樿浠垮皠鐭╅樀
            affine = np.eye(4)
        
        # 鍒涘缓NIfTI鍥惧儚瀵硅薄
        nii_img = nib.Nifti1Image(data, affine)
        
        # 淇濆瓨锛堣嚜鍔ㄥ鐞?gz鍘嬬缉锛?
        nib.save(nii_img, filepath)
        return True
    
    def _save_nrrd(self, data: np.ndarray, filepath: str, metadata: Optional[Dict]) -> bool:
        """淇濆瓨NRRD鏂囦欢"""
        # 杞崲涓篠impleITK鏍煎紡 (D, H, W)
        data_transposed = np.transpose(data, (2, 0, 1))
        sitk_image = sitk.GetImageFromArray(data_transposed)
        
        # 璁剧疆绌洪棿鍙傛暟
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
        
        # 淇濆瓨
        sitk.WriteImage(sitk_image, filepath)
        return True
    
    def load_label_map(
        self,
        filepath: str,
        reference_metadata: Optional[Dict] = None,
        reference_shape: Optional[Tuple[int, int, int]] = None,
        force_align: bool = True,
    ) -> Optional[np.ndarray]:
        """
        加载标签掩码文件。

        说明：即使标签数组 shape 与当前影像一致，也优先做几何对齐。
        不同来源文件可能存在方向差异（如 LPS/RAS、左右翻转），
        若直接按数组索引叠加，会出现镜像或偏移。
        """
        ref_meta = reference_metadata if isinstance(reference_metadata, dict) else {}
        ref_sitk = ref_meta.get("sitk_image", None)

        # 先读取原始标签（作为最终回退）
        raw_data, _ = self.load_image(filepath)
        if raw_data is not None:
            raw_data = raw_data.astype(np.uint8)
            if not force_align:
                raw_shape_ok = (
                    reference_shape is None
                    or tuple(raw_data.shape) == tuple(reference_shape)
                )
                if raw_shape_ok:
                    return raw_data

        aligned_data = None

        # 1) 优先使用 SimpleITK 参考几何做最近邻重采样
        if ref_sitk is not None:
            try:
                label_img = sitk.ReadImage(filepath)
                label_img = sitk.Cast(label_img, sitk.sitkUInt8)
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(ref_sitk)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler.SetDefaultPixelValue(0)
                aligned = resampler.Execute(label_img)
                data = sitk.GetArrayFromImage(aligned)
                data = np.transpose(data, (1, 2, 0)).astype(np.uint8)
                aligned_data = data
            except Exception as e:
                print(f"Label SITK对齐失败，回退常规读取: {e}")

        # 2) 次选：若有参考 affine，使用 nibabel 最近邻重采样
        if aligned_data is None:
            ref_affine = ref_meta.get("affine", None)
            if reference_shape is not None and ref_affine is not None and resample_from_to is not None:
                try:
                    ref_affine = np.asarray(ref_affine, dtype=np.float64)
                    if ref_affine.shape == (4, 4) and np.all(np.isfinite(ref_affine)):
                        src = nib.load(filepath)
                        aligned = resample_from_to(src, (tuple(reference_shape), ref_affine), order=0)
                        data = aligned.get_fdata()
                        if data is not None:
                            aligned_data = np.asarray(data, dtype=np.uint8)
                except Exception as e:
                    print(f"Label affine对齐失败，回退常规读取: {e}")

        # 若得到对齐结果，默认返回；当对齐后体素量明显异常时，允许回退原始数组
        if aligned_data is not None:
            use_aligned = True
            if raw_data is not None:
                raw_nz = int(np.count_nonzero(raw_data))
                aligned_nz = int(np.count_nonzero(aligned_data))
                if raw_nz > 0:
                    ratio = aligned_nz / float(raw_nz)
                    if ratio < 0.35 or ratio > 3.0:
                        # 若原始标签尺寸可直接使用，则优先回退原始数组，避免异常几何对齐导致形态畸变。
                        raw_shape_ok = (
                            reference_shape is None
                            or tuple(raw_data.shape) == tuple(reference_shape)
                        )
                        if raw_shape_ok:
                            use_aligned = False
                            print(
                                f"Label对齐后体素量异常(raw={raw_nz}, aligned={aligned_nz}, ratio={ratio:.3f})，"
                                "已回退原始标签数组。"
                            )
                        else:
                            print(
                                f"Label对齐后体素量异常(raw={raw_nz}, aligned={aligned_nz}, ratio={ratio:.3f})，"
                                "因原始标签尺寸不匹配，仍采用对齐结果。"
                            )
                elif aligned_nz == 0:
                    print("Label原始标签与对齐标签均为空。")
            if use_aligned:
                return aligned_data

        # 回退：无法对齐时返回原始读取结果
        return raw_data


    def save_label_map(self, label_data: np.ndarray, filepath: str, metadata: Optional[Dict] = None) -> bool:
        """
        淇濆瓨鏍囩鎺╃爜鏂囦欢
        
        Args:
            label_data: 鏍囩鏁版嵁 (H, W, D)锛屽簲涓簎int8绫诲瀷
            filepath: 淇濆瓨璺緞
            metadata: 鍏冩暟鎹紙搴斾笌鍘熷褰卞儚涓€鑷达級
            
        Returns:
            鎴愬姛杩斿洖True锛屽け璐ヨ繑鍥濬alse
        """
        # 纭繚涓簎int8绫诲瀷
        if label_data.dtype != np.uint8:
            label_data = label_data.astype(np.uint8)
        
        return self.save_image(label_data, filepath, metadata)



