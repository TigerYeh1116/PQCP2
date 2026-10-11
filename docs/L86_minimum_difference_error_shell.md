# 固定支撐的最小非零差分誤差：能量 4 檢查

本文件使用主搜尋的 antipodal C/G 編碼。
它提供精確有限搜尋及固定支撐的能量下界；不把近似解算成 PQCP。

## 1. 為什麼只需要 420 個目標

從 exact C witness 開始，其支撐已滿足目標的全部 mod-2 PACF。
G 是互補支撐，兩列 G 的總重量仍為 43。
對非零位移 u，P_{1-U}(u)=43-2|U|+P_U(u)。
兩列相加且 |U_A|+|U_B|=43，得到 G unsigned PACF 與 C unsigned PACF 相同。
C/G targets 的 ±1 號可能不同，但 parity 相同。
因此任意 G sign 配置與 G target 的每個獨立位移誤差 ε_u 都是偶數。

contents 固定時，每列 ΣG=o-e，與 sign 的位置無關。
恆等式 Σ_u P_G(u)=(ΣG)² 因而強制

\[
\sum_{u=1}^{21}\epsilon_u=0.
\]

若 ε 不全為零，最小可能的 Σ ε_u² 是 8，
只能是一個 +2、一個 -2，分別在兩個不同的獨立位移。
共有 21×20=420 個有向配置。
每個配置都擴張成對稱的 43 點目標，再交給完整 native sign solver。
不是只對 420 個符號配置抽樣，而是逐個目標求解它的全部符號可行性。

當 C error 為零，原始完整能量 E_full=(E_C+E_G)/2=4。
任何 SAT 都按原始長度重建並核對全部整數 PACF、parity contents 及 E_full=4。
它仍是非零誤差候選，不能當作新 PQCP 或發現速度證據。

## 2. 完整無解能提供什麼下界

若 G exact-zero 已完整 UNSAT，而且上述 420 個 profiles 也全部完整 UNSAT，
G 不可能有 squared error 0 或 8。
寫 ε=2z，Σz=0，Σz² 必為偶數，且最小非零值為 2。
排除這個值後，Σz²≥4，所以 E_G≥16，E_full≥8。
這個下界涵蓋同一支撐／contents 的全部 C/G signs，因為 E_C≥0；
它不需要把已找到的某個 C sign witness 當成唯一 C 配置。
任何 UNKNOWN 都不允許宣稱此下界。

## 3. 已完成實驗與測試

原長測的 9 個支撐，各自 G zero 與全部 420 profiles 均完整 UNSAT，
共 3,780 個最小誤差目標，沒有 UNKNOWN、沒有 energy-4 candidate。
記錄 `results/L86_original_coarse_near_G_shell_20261007/report.json`。
新 m=20 區塊產生的 20 個支撐，同樣全部 8,400 個 profiles 完整 UNSAT，
沒有 UNKNOWN、沒有 energy-4 candidate。
記錄 `results/L86_new_coarse_near_G_shell_20261007/report.json`。
因此這 29 個支撐在對應 contents 下都無法把完整能量壓到 8 以下。
這是固定支撐的下界，不是一般 L86 全域下界。

測試以小長度的完整 even-error Cartesian cube 驗證 profile family 沒有遺漏，
並用一個 literal L14 C/G witness 核對重建後的完整 binary energy 恰為 4、但不是 PQCP。
這個正例驗證 SAT candidate 分支的數學，不只測試 L86 負例。

實作：`solver/near_difference_profiles.py`、`experiments/search_near_difference_shell.py`。
IDE 可直接執行該實驗入口，預設檢查原長測 9 組 C supports，
結果保存於帶時間戳的獨立 results 目錄。
