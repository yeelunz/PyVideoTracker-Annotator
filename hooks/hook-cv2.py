# Hook for cv2 to fix circular import and configuration issues
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
import cv2
import os

# Collect all cv2 submodules
hiddenimports = collect_submodules('cv2')

# Collect all data files including config files
datas = collect_data_files('cv2', include_py_files=True)

# Collect dynamic libraries
binaries = collect_dynamic_libs('cv2')

# Get cv2 installation path and add config files
try:
    cv2_path = os.path.dirname(cv2.__file__)
    config_files = []
    for root, dirs, files in os.walk(cv2_path):
        for file in files:
            if file.endswith(('.xml', '.yml', '.yaml', '.json', '.cfg', '.config', '.py')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, cv2_path)
                config_files.append((full_path, os.path.join('cv2', rel_path)))
    
    datas.extend(config_files)
except:
    pass

# Additional hidden imports to resolve circular dependencies
hiddenimports += [
    'cv2.cv2',
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy._distributor_init',
    'numpy.core.multiarray',
    'numpy.core.umath',
]
