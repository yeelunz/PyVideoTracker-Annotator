# Hook for cv2 to fix circular import and configuration issues
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs
import cv2
import os
import glob

# Collect all cv2 submodules
hiddenimports = collect_submodules('cv2')

# Collect data files but avoid including .py files as data - the compiled
# extension (.pyd) is used at runtime. Including .py files as datas can
# create duplicate/nested entries that break onefile extraction.
datas = collect_data_files('cv2', include_py_files=False)

# Collect dynamic libraries. Rather than relying solely on collect_dynamic_libs
# (which may return nested targets), explicitly locate the key binaries
# (.pyd and opencv_*.dll) and map them to the top-level destination so that
# onefile extraction will place them where the interpreter can import/load them.
binaries = []
try:
    cv2_path = os.path.dirname(cv2.__file__)
    # collect pyds in cv2 root (e.g. cv2.cp311-win_amd64.pyd or cv2.pyd)
    for p in glob.glob(os.path.join(cv2_path, '*.pyd')):
        binaries.append((p, '.'))
    # collect common opencv DLLs (ffmpeg, opencv_world, etc.) under cv2
    for p in glob.glob(os.path.join(cv2_path, '**', 'opencv*.dll'), recursive=True):
        binaries.append((p, '.'))
    # also try to collect ffmpeg dll (some wheels provide opencv_videoio_ffmpeg*.dll)
    for p in glob.glob(os.path.join(cv2_path, 'opencv_videoio_*.dll')):
        binaries.append((p, '.'))
except Exception:
    # Fallback: allow collect_dynamic_libs to attempt discovery
    binaries += collect_dynamic_libs('cv2')

# Get cv2 installation path and add config files
try:
    cv2_path = os.path.dirname(cv2.__file__)
    config_files = []
    # Place resource/config files under a dedicated top-level folder `cv2_data`
    # to avoid collisions with any file named "cv2" that PyInstaller might
    # create during collection (which would make a directory creation fail).
    for root, dirs, files in os.walk(cv2_path):
        for file in files:
            # only include non-.py config/resource files (xml/yml/json/etc.)
            if file.endswith(('.xml', '.yml', '.yaml', '.json', '.cfg', '.config')):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, cv2_path)
                # Normalize and guard against rel_path being '.' or starting with '..'
                if rel_path in ('.', '') or rel_path.startswith('..'):
                    rel_path = file
                dest = os.path.join('cv2_data', rel_path.replace('\\', '/'))
                config_files.append((full_path, dest))

    datas.extend(config_files)
except Exception:
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

# Deduplicate datas and binaries (some packaging runs can collect duplicates
# from different discovery paths). Using dict.fromkeys on the tuples preserves
# order while removing exact duplicates.
try:
    datas = list(dict.fromkeys(datas))
except Exception:
    pass

try:
    binaries = list(dict.fromkeys(binaries))
except Exception:
    pass
