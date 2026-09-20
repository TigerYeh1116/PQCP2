"""Click Run in VS Code: compile and launch the independent C verifier.

All mathematical checks remain in verify_pqcp.c. This standard-library-only
launcher always uses the directory containing this file for L.txt lookup.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> int:
    """Build safely, inherit terminal input/output, and propagate verification status."""
    root = Path(__file__).resolve().parent
    compiler = shutil.which("cc")
    if compiler is None:
        print("找不到 C 編譯器 cc。macOS 請先安裝 Xcode Command Line Tools "
              "（xcode-select --install）。", file=sys.stderr)
        return 2
    try:
        build = root / "build"
        build.mkdir(exist_ok=True)
        executable = build / "verify_pqcp"
        # Rebuild on every launch: never run a stale verifier after a source edit.
        # A failed compilation leaves the previous executable untouched.
        with tempfile.TemporaryDirectory(prefix="verify_pqcp_", dir=build) as temporary:
            compiled = Path(temporary) / "verify_pqcp"
            subprocess.run(
                [compiler, "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
                 str(root / "verify_pqcp.c"), "-o", str(compiled)],
                check=True,
            )
            os.replace(compiled, executable)
        return subprocess.run(
            [str(executable), *sys.argv[1:]], cwd=root,
        ).returncode
    except subprocess.CalledProcessError:
        print("C 驗證程式編譯失敗，未執行驗證。", file=sys.stderr)
        return 2
    except OSError as error:
        print("無法啟動 C 驗證程式：{}".format(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n驗證已中止，未修改結果檔案。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
