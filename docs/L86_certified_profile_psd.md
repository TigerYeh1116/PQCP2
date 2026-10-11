# 有理數上界的 PSD 不可能證書

這是本專案自行實作的嚴格必要條件。對支撐 U 的 intersection profile N，
循環自相關的 Fourier transform 是 |Û(j)|²，因此必须非負。
對稱 profile 的值為

\[
F_j=N(0)+2\sum_{u=1}^{21}N(u)\cos(2\pi ju/43).
\]

若有一個頻率的嚴格上界小於零，即可證明 profile 不可能。
這不使用 FFT 誤差、epsilon 或浮點符號判定。

## 全部係數向外取整

使用 Machin 恆等式 π=16 atan(1/5)-4 atan(1/239)，
各 atan 的 20 項交錯有理級數及下一項提供 π_lower、π_upper。
程式用 Fraction 核對 tangent 恆等式與 3<π_lower<π_upper<22/7。

把 phase 折至 0..21，令 x=2π_lower phase/43，真角度在 [x,π)。
cos 在這個區間遞減。cos(x) 的偶數截斷
Σ_{n=0}^{16}(-1)^n x^{2n}/(2n)! 是上界；
π_upper²<12 保證相關交錯尾項遞減。
令 S=2⁴⁰，計算 C_phase=ceil(S·該有理截斷)，則
C_phase/S 必定大於等於真 cosine。

N(u) 已先核對非負，因此

\[
F_j\le [N(0)S+2\sum_uN(u)C_{ju}]/S.
\]

分子嚴格負時輸出整數證書：profile、頻率、全部 coefficient numerators、
分母 S 及負分子。零／正上界只表示未被此必要條件排除。
`obstruction_only=True` 的通過結果是 complete=False，不能當成 SAT 或 UNSAT。

## 配對結果及限制

`results/L86_half_row_psd_paired_20261007/report.json`：
同一批 65 個中心、每回放 260 個 fibers，三次暖機配對，交替執行順序。
全部 periodic masks、恢復 phases、完整模型集合一致；
中位回放時間 1.194687209 → 0.782945417 秒，局部减少 34.4644%。
必要 norm visits 93,977,532 → 2,247,696。

兩邊完整新 L86 等價類都是零，這不是「新解發現速度提升 20%」證據。
目前只證明這批完整子空間的成本降低，尚未證明整個搜尋 pipeline 的改善。
測試包括全部 L11 binary profiles、零／全一的 PSD 邊界，及高精度係數上界核對。
程式：`solver/certified_profile_psd.py`。
