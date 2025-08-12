# Hook for numpy to fix DLL loading issues
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

# Collect all numpy submodules
hiddenimports = collect_submodules('numpy')

# Collect data files and dynamic libraries
datas = collect_data_files('numpy', include_py_files=False)
binaries = collect_dynamic_libs('numpy')

# Additional hidden imports
hiddenimports += [
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy._distributor_init',
    'numpy.core.multiarray',
    'numpy.core.umath',
    'numpy.linalg.lapack_lite',
    'numpy.linalg._umath_linalg',
]
