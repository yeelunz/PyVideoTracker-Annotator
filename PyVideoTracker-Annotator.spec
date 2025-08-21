# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
import os
import importlib

# Dynamically locate cv2 installation path instead of a hard-coded absolute path
datas = []
binaries = []
try:
    cv2 = importlib.import_module('cv2')
    cv2_path = os.path.dirname(cv2.__file__)
    # Note: do not append the whole package directory here; rely on
    # collect_data_files('cv2') below to gather package data. Appending
    # the directory directly can create duplicate entries and nested
    # targets which lead to extraction errors at runtime.
except Exception:
    # Fall back to collect_data_files which will try to find package data
    pass

# Note: hook-cv2.py in the project's hooks folder will collect cv2's
# data and dynamic libraries. Avoid collecting them again here to
# prevent duplicate entries which can produce nested paths and
# extraction failures in onefile builds.


a = Analysis(
    ['main.py'],
    pathex=[r'C:\Users\asas1\PyVideoTracker-Annotator'],
    binaries=binaries,
    datas=datas,
    hiddenimports=['cv2', 'numpy.core._methods', 'numpy.lib.format'],
    hookspath=[r'C:\Users\asas1\PyVideoTracker-Annotator\hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['pygame','pygame.*','tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='PyVideoTracker-Annotator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Create a COLLECT bundle (onedir). This makes the spec produce a folder
# with all binaries and data expanded, which is easier to inspect for missing
# cv2 .pyd/.dll files compared to onefile mode.
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='PyVideoTracker-Annotator'
)
