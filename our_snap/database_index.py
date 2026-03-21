#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据库索引模块（轻量级）

目标：
- 支持扫描类似 522/data/.../ST*/SE* 这样的目录结构
- 找出“可直接加载”的 DICOM 序列目录（即包含 IM* 这类无后缀切片文件的目录）
- 为 UI 提供一个可浏览/可检索的索引列表
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

import SimpleITK as sitk


@dataclass(frozen=True)
class SeriesEntry:
    """一个可加载的序列（细化到 DICOM SeriesInstanceUID 粒度）"""

    series_dir: str
    patient_id: str
    study_id: str
    series_id: str
    series_uid: Optional[str] = None

    # 可选的 DICOM 元信息
    patient_name: Optional[str] = None
    study_date: Optional[str] = None
    modality: Optional[str] = None
    series_description: Optional[str] = None
    file_list: Optional[List[str]] = None  # 构成该序列的 DICOM 文件列表

    @property
    def display_name(self) -> str:
        """用于列表显示的简要描述"""
        parts = [self.patient_id, self.study_id, self.series_id]
        extra: List[str] = []
        if self.modality:
            extra.append(self.modality)
        if self.study_date:
            extra.append(self.study_date)
        if self.series_description:
            extra.append(self.series_description)
        if extra:
            parts.append(" - " + " | ".join(extra))
        return " / ".join(parts)

    @property
    def tooltip_text(self) -> str:
        """用于鼠标悬停时的完整信息"""
        lines = [
            f"路径: {self.series_dir}",
            f"Patient ID: {self.patient_id}",
        ]
        if self.patient_name:
            lines.append(f"Patient Name: {self.patient_name}")
        lines.append(f"Study ID: {self.study_id}")
        if self.study_date:
            lines.append(f"Study Date: {self.study_date}")
        lines.append(f"Series ID: {self.series_id}")
        if self.modality:
            lines.append(f"Modality: {self.modality}")
        if self.series_description:
            lines.append(f"Series Description: {self.series_description}")
        return "\n".join(lines)


def _is_probably_im_file(filename: str) -> bool:
    """
    判断文件名是否形如 IM0 / IM1 / IM23（无扩展名）
    """
    if not filename.startswith("IM"):
        return False
    suffix = filename[2:]
    return suffix.isdigit() and len(suffix) > 0


def _series_dir_has_im_files(series_dir: str) -> bool:
    """
    判断目录下是否存在 IM* 文件
    """
    try:
        for name in os.listdir(series_dir):
            p = os.path.join(series_dir, name)
            if os.path.isdir(p):
                continue
            if _is_probably_im_file(name):
                return True
        return False
    except Exception:
        return False


def _get_first_im_file(series_dir: str) -> Optional[str]:
    """
    返回序列目录中第一个 IM* 文件路径，用于读取 DICOM 元数据
    """
    try:
        names = sorted(os.listdir(series_dir))
        for name in names:
            p = os.path.join(series_dir, name)
            if os.path.isdir(p):
                continue
            if _is_probably_im_file(name):
                return p
    except Exception:
        return None
    return None


def _read_dicom_metadata(example_file: str) -> dict:
    """
    用 SimpleITK 从一张切片中读取常用 DICOM 标签
    """
    meta: dict = {}
    if not example_file:
        return meta

    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(example_file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        def get_tag(tag: str) -> Optional[str]:
            return reader.GetMetaData(tag) if reader.HasMetaDataKey(tag) else None

        meta["patient_name"] = get_tag("0010|0010")
        meta["patient_id"] = get_tag("0010|0020")
        meta["study_date"] = get_tag("0008|0020")
        meta["modality"] = get_tag("0008|0060")
        meta["series_description"] = get_tag("0008|103e")
        meta["study_id_tag"] = get_tag("0020|0010")
        meta["series_number"] = get_tag("0020|0011")
    except Exception:
        # 元数据读取失败时静默返回空字典，避免影响整体扫描
        pass
    return meta


def _split_ids_from_path(series_dir: str) -> Optional[tuple[str, str, str]]:
    """
    尝试从路径末端解析 patient/study/series:
      .../<patient>/ST0/SE2
    """
    parts = os.path.normpath(series_dir).split(os.sep)
    if len(parts) < 3:
        return None
    series_id = parts[-1]
    study_id = parts[-2]
    patient_id = parts[-3]
    return patient_id, study_id, series_id


def build_522_database_index(
    db_root: str,
    stop_checker: Optional[Callable[[], bool]] = None,
) -> List[SeriesEntry]:
    """
    扫描数据库根目录，返回可加载序列列表。

    典型根目录示例：
      E:\\ITK-SNAP 4.2\\522\\data
    """
    db_root = os.path.abspath(db_root)
    entries: List[SeriesEntry] = []

    # 经验上 DICOM 序列通常在 .../<patient>/ST*/SE* 之类的叶子目录中。
    # 为了避免扫描整个树带来的巨大开销和大量 GDCM warning：
    # - 仅在「目录名以 SE 开头」的目录上尝试读取 SeriesUID
    # - 并且要求目录里存在 IM* 文件（快速过滤）
    for dirpath, dirnames, _filenames in os.walk(db_root):
        if stop_checker and stop_checker():
            break

        base = os.path.basename(dirpath)
        if not base.startswith("SE"):
            continue
        if not _series_dir_has_im_files(dirpath):
            continue

        reader = sitk.ImageSeriesReader()
        try:
            series_ids = reader.GetGDCMSeriesIDs(dirpath)
        except Exception:
            series_ids = []

        # 如果 GDCM 没返回任何 series，但目录里有 IM* 切片，则按单序列兜底
        manual_series = False
        manual_files: list[str] = []
        if not series_ids:
            names = sorted(os.listdir(dirpath))
            manual_files = [
                os.path.join(dirpath, n)
                for n in names
                if os.path.isfile(os.path.join(dirpath, n)) and _is_probably_im_file(n)
            ]
            if manual_files:
                manual_series = True
                series_ids = ["manual_series"]  # 放一个占位 UID

        # 从目录路径解析一个粗略的 patient/study/series 结构
        ids = _split_ids_from_path(dirpath)
        if ids is None:
            patient_hint, study_hint, series_hint = "", "", ""
        else:
            patient_hint, study_hint, series_hint = ids

        for uid in series_ids:
            if stop_checker and stop_checker():
                break
            if manual_series:
                file_names = manual_files
            else:
                try:
                    file_names = reader.GetGDCMSeriesFileNames(dirpath, uid)
                except Exception:
                    file_names = []
            if not file_names:
                continue

            # 从该 series 的第一张切片读取元信息
            meta = _read_dicom_metadata(file_names[0])

            # 优先使用 DICOM 中的 PatientID / StudyID / SeriesNumber（如果存在）
            patient_id_meta = meta.get("patient_id") or patient_hint
            study_id_meta = meta.get("study_id_tag") or study_hint
            series_id_meta = meta.get("series_number") or series_hint or uid

            entry = SeriesEntry(
                series_dir=dirpath,
                patient_id=patient_id_meta,
                study_id=study_id_meta,
                series_id=series_id_meta,
                series_uid=None if manual_series else uid,
                patient_name=meta.get("patient_name"),
                study_date=meta.get("study_date"),
                modality=meta.get("modality"),
                series_description=meta.get("series_description"),
                file_list=file_names,
            )
            entries.append(entry)

        # SE* 下面通常没有再需要递归的内容，直接剪枝
        dirnames[:] = []

    # 稳定排序，方便 UI 展示
    entries.sort(key=lambda e: (e.patient_id, e.study_id, e.series_id, e.series_dir))
    return entries


def find_patient_dirs_522(db_root: str) -> List[str]:
    """
    在 522/data 根目录下找到“病人目录”列表（尽量快，不读DICOM）。

    常见结构：
      <db_root>/<range>/<patient>/ST*/SE*/IM*
    """
    db_root = os.path.abspath(db_root)
    patient_dirs: List[str] = []
    try:
        ranges = [os.path.join(db_root, d) for d in os.listdir(db_root)]
    except Exception:
        return patient_dirs

    for r in ranges:
        if not os.path.isdir(r):
            continue
        try:
            patients = [os.path.join(r, d) for d in os.listdir(r)]
        except Exception:
            continue
        for p in patients:
            if not os.path.isdir(p):
                continue
            # 快速判断：至少存在一个 ST* 子目录
            try:
                if any(name.startswith("ST") and os.path.isdir(os.path.join(p, name)) for name in os.listdir(p)):
                    patient_dirs.append(p)
            except Exception:
                continue

    patient_dirs.sort()
    return patient_dirs


def build_patient_index_522(
    patient_dir: str,
    stop_checker: Optional[Callable[[], bool]] = None,
) -> List[SeriesEntry]:
    """
    仅扫描某个病人目录（比扫全库快很多），返回该病人下所有可用的 SeriesUID 条目。
    """
    return build_522_database_index(patient_dir, stop_checker=stop_checker)


