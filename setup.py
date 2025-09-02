import platform
import runpy
import sys

os_name = platform.system().lower()

if os_name == 'linux':
    runpy.run_path("setup_linux.py", run_name="__main__")
elif os_name == 'darwin':
    runpy.run_path("setup_macos.py", run_name="__main__")
else:
    print(f"Unsupported OS: {os_name}", file=sys.stderr)
    sys.exit(1)