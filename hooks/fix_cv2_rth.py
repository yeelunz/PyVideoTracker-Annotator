import sys
import os

# Runtime hook to adjust permissions of extracted cv2 files in onefile mode.
# PyInstaller extracts bundled files into sys._MEIPASS; some files (e.g. config.py)
# have triggered PermissionError when OpenCV tries to open them. This hook makes
# sure those files are readable and not locked.
try:
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        # target common places where cv2 hook may have placed files
        candidates = [os.path.join(meipass, 'cv2'), os.path.join(meipass, 'cv2_data')]
        for base in candidates:
            if os.path.isdir(base):
                for root, dirs, files in os.walk(base):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            # ensure owner read/write and group/other read
                            os.chmod(fp, 0o666)
                        except Exception:
                            # ignore permission errors and continue
                            pass
        # also try top-level config.py just in case
        top_cfg = os.path.join(meipass, 'config.py')
        if os.path.exists(top_cfg):
            try:
                os.chmod(top_cfg, 0o666)
            except Exception:
                pass
except Exception:
    pass
