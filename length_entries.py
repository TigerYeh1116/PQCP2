"""Menus behind the per-length IDE entries L44.py, L46.py, L68.py, L86.py, L94.py.

Each entry is a thin file whose Run button calls one function here. Search
code stays in main.py and experiments/; this module only asks the few
questions needed and launches the matching program with this interpreter.

There are no entries for L=58 and L=90: no (L,4)-PQCP exists there (see
pqcp_length_obstruction). L=94 passes that test; its existence is open.
"""

from math import isqrt
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SAMPLING_TRIALS_PER_SECOND = 8400  # L86 weight-focused sampling, 10 workers, measured 2026-10-21.


def _sum_of_two_squares(n):
    return n >= 0 and any(isqrt(n-a*a)**2 == n-a*a for a in range(isqrt(n)+1))


def pqcp_length_obstruction(length):
    """Reason no (L,4)-PQCP exists for even L, or None if the necessary test passes.

    With A(x)A(1/x)+B(x)B(1/x) = 2L+4*eta*(x^k+x^-k), put x=1 and x=-1. Row and
    alternating sums of a +-1 word of even length are even, so dividing by 4:
    L/2+2*eta and L/2+2*eta*(-1)^k must both be sums of two squares, 1<=k<L/2.
    """
    half = length//2
    if length % 2 or half < 2:
        return "需要偶數 L≥4（1≤k<L/2 才有位移可用）"
    for eta in (1, -1):
        for k in (1, 2):
            if k < half and _sum_of_two_squares(half+2*eta) and _sum_of_two_squares(half+2*eta*(-1)**k):
                return None
    return "{}±2 = {}、{} 不符合必要條件（x=±1 代入後須為兩平方和），此長度無解".format(
        half, half+2, half-2)


def _ask_choice(title, options, default):
    """options: (key, label, action) rows; Enter selects default."""
    print(title)
    for key, label, _ in options:
        print("  {}. {}".format(key, label))
    actions = {key: action for key, _, action in options}
    while True:
        choice = input("請選擇（直接按 Enter = {}）：".format(default)).strip() or default
        if choice in actions:
            return actions[choice]
        print("請輸入選單上的數字。")


def _ask_positive_int(prompt, default):
    while True:
        raw = input(prompt).strip().replace(",", "").replace("_", "")
        if not raw:
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("請輸入正整數，或直接按 Enter 使用預設值。")


def _run_script(script, *arguments):
    """Run another program with this interpreter; Ctrl+C reaches it through the terminal."""
    process = subprocess.Popen([sys.executable, str(ROOT/script), *map(str, arguments)], cwd=ROOT)
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            continue  # The child received the same Ctrl+C and saves its own resume state.


def _run_pytorch(length):
    import main  # Heavy imports (torch) only when the PyTorch flow is chosen.
    return main.main(preset_length=length)


def _common_actions(length):
    return (
        ("1", "PyTorch 主流程（L={}）".format(length), lambda: _run_pytorch(length)),
        ("2", "驗證 {}.txt（獨立 C 驗證器）".format(length), lambda: _run_script("verify_pqcp.py", length)),
        ("3", "檢查 {}.txt 是否有等價重複（只預覽，不修改）".format(length),
         lambda: _run_script("experiments/deduplicate_solutions.py", "--L", length)),
        ("0", "離開", lambda: 0),
    )


def length_entry(length):
    """Entry for one Project 2 length searched by the PyTorch pipeline."""
    return _ask_choice("L={} 入口".format(length), _common_actions(length), "1")()


def l44_entry():
    """L44 entry; short lengths below 44 (e.g. 4) are merged here, impossible ones refused."""
    while True:
        raw = input("長度（直接按 Enter = 44；也可輸入 44 以下的短長度，例如 4）：").strip()
        if not raw:
            length = 44
            break
        if not (raw.isdigit() and 4 <= int(raw) <= 44):
            print("請輸入 4～44 的偶數。")
            continue
        reason = pqcp_length_obstruction(int(raw))
        if reason is None:
            length = int(raw)
            break
        print("L={}：{}。".format(raw, reason))
    return length_entry(length)


def _l86_sampling():
    trials = _ask_positive_int("抽樣筆數（直接按 Enter = 80,000,000，約 2.6 小時）：", 80_000_000)
    print("預估約 {:.1f} 小時（10 核）；中途可按 Ctrl+C，之後選 2 續跑。".format(
        trials/SAMPLING_TRIALS_PER_SECOND/3600))
    return _run_script("experiments/run_l86_test_campaign.py", "--no-frontier", "--sampler-trials", trials)


def _l86_resume():
    unfinished = sorted(p for p in (ROOT/"results").glob("L86_test_campaign_*")
                        if (p/"plan.json").exists() and not (p/"campaign_report.json").exists())
    if not unfinished:
        print("沒有找到中斷的 L86 抽樣。")
        return 0
    print("續跑：{}".format(unfinished[-1].name))
    return _run_script("experiments/run_l86_test_campaign.py", "--resume", unfinished[-1])


def l94_entry():
    """L94 entry; the low-energy truncated search is the default method."""
    common = {key: (label, action) for key, label, action in _common_actions(94)}
    options = (
        ("1", "低能量截斷搜尋找 L94（新方法，長跑，按 Ctrl+C 停；新解驗證後寫入 94.txt）",
         lambda: _run_script("experiments/search_l94_truncated.py")),
        ("2", "GPU+CPU 支撐抽樣（舊方法，約慢 12 倍）", lambda: _run_script("experiments/search_l94_supports.py")),
        ("3", common["1"][0], common["1"][1]),
        ("4", common["2"][0], common["2"][1]),
        ("5", common["3"][0], common["3"][1]),
        ("0", "離開", lambda: 0),
    )
    print("L94 已找到 3 組；新方法（GPU 掃描）平均約 2 小時找到下一組（4 小時內約 85% 機會）。")
    return _ask_choice("L=94 入口", options, "1")()


def l86_entry():
    """L86 entry; weight-focused sampling is the productive method and the default."""
    common = {key: (label, action) for key, label, action in _common_actions(86)}
    options = (
        ("1", "加權抽樣找新解（w=19～24，新解驗證後自動寫入 86.txt）", _l86_sampling),
        ("2", "續跑中斷的加權抽樣", _l86_resume),
        ("3", common["1"][0]+"——L86 上歷來沒有產出，不建議", common["1"][1]),
        ("4", "全域 SAT（不固定 C_A）", lambda: _run_script("experiments/run_l86_global_sat.py")),
        ("5", "weight-group 舊版抽樣", lambda: _run_script("experiments/run_l86_weight_group_search.py")),
        ("6", common["2"][0], common["2"][1]),
        ("7", common["3"][0], common["3"][1]),
        ("0", "離開", lambda: 0),
    )
    return _ask_choice("L=86 入口", options, "1")()
