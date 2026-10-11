# L86：模 8 二次布林剪枝與精確提升

## 文獻依據與本專案的推導

2026 年刊出的 [Lumsden、Kotsireas、Bright：New Results on Periodic Golay Pairs](https://arxiv.org/html/2408.15611v3)
使用 compression、orderly generation 及 multilevel uncompression。
這支持「先解壓縮約束、再精確提升」的組織方式。
但 Golay 的非零位移目標為零，不能將該論文的條件原樣套用本專案的兩個 ±4 旁瓣。

2026 年 [Andraschko 等人的 ICMS 摘要](https://icms-conference.org/2026/abstracts/Andraschko.pdf)
討論以 SAT 求 periodic Golay pairs 及壓縮／提升；此處參考的是公開摘要，並未取得完整論文。

2025 年 [Joux、Mora：The regular multivariate quadratic problem](https://link.springer.com/article/10.1007/s10623-025-01717-6)
分析有限域二次方程與額外組合約束的求解方式，包含 Boolean 系統的 Gray-code 枚舉。
該文的 regular constraints 並非本專案的 parity content；不能引用其複雜度作為 PQCP 的加速保證。

以下「PACF 模 8 轉換」及其與本專案符號求解器的銜接是本專案自行推導。
實作使用確定性的完整枚舉；沒有引用論文中的隨機隔離作為排除依據。

## 1. 從精確 PACF 到二次 Boolean 方程

固定奇數長度 d 的兩個 supports，令 active 符號為
\(a_i=(-1)^{s_i}\)，\(b_i=(-1)^{t_i}\)，其他位置為零。
位移 u 上共有 \(M_u\) 個 active directed edges。
令 \(D_u\) 為其中端點符號不同的 edges 數。
則

\[
P_a(u)+P_b(u)=M_u-2D_u.
\]

目標 \(C_u\) 給出精確要求
\(D_u=r_u=(M_u-C_u)/2\)。
既有 GF(2) 系統只強制 \(D_u\equiv r_u\pmod2\)，即 PACF 模 4。
Gaussian elimination 後，所有符號位元都是自由變數 f 的 affine Boolean 函數。
每條 edge 的 mismatch \(z_e(f)=s_i\mathbin\oplus s_j\) 也是 affine。

對任何 Boolean bits \(z_1,\ldots,z_m\)，在 GF(2) 中

\[
\left\lfloor\frac{\sum_e z_e}{2}\right\rfloor\bmod2
=\binom{\sum_e z_e}{2}\bmod2
=\bigoplus_{e<e'} z_e z_{e'}.
\]

第一個等式可用整數總和模 4 的四種情況直接驗證。
第二個等式是數出同時為 1 的 bit pairs。
將 affine 函數代入並使用 \(f_i^2=f_i\)，右側的 Boolean 代數次數最多為二。
所以增加方程

\[
Q_u(f)=\operatorname{bit}_1(D_u(f))
\mathbin\oplus\operatorname{bit}_1(r_u)=0
\]

恰好強制 \(D_u\equiv r_u\pmod4\)，也就是 PACF 模 8。
對每列的負號個數也使用相同公式，加入指定負號數模 4 的必要條件。
L43 有 21 個 independent shifts，加兩列計數，共 23 個 Boolean 方程。
這些條件必要但不充分；通過後仍要做完整整數 PACF 及精確符號計數。

## 2. 不靠猜測取得全部二次係數

將 23 個方程的值打包成一個 uint32 syndrome。
因為次數至多二，整個 vector-valued 函數唯一寫成

\[
Q(f)=c\oplus\bigoplus_i\ell_i f_i
\oplus\bigoplus_{i<j}q_{ij}f_if_j.
\]

直接整數計算在 0、每個單位向量及每對單位向量相加處的值即可得到

\[
c=Q(0),\quad \ell_i=Q(e_i)\oplus c,
\quad q_{ij}=Q(e_i+e_j)\oplus c\oplus\ell_i\oplus\ell_j.
\]

這不是以少量樣本猜測一般函數，而是在已證明二次的函數空間內取得精確係數。

## 3. Gray-code 完整枚舉

當 f 的第 i 個 bit 翻轉時，

\[
Q(f+e_i)=Q(f)\oplus\ell_i
\oplus\bigoplus_{j\ne i}q_{ij}f_j.
\]

將 f 分成兩半，預先列出每個 i 在兩半上的線性 derivative table。
每步用兩次 table lookup 及 XOR 更新 syndrome。
只有零 syndrome 才需要精確負號數與完整 PACF 檢查。
仍枚舉原 affine space 的每個 assignment，沒有移除任意 parity 可行解。

實作最多處理 24 個自由變數，兩個 table 合計最多約 0.75 MiB，呼叫結束即釋放。
上限以外使用原 kernel；預算未完整覆蓋時回 UNKNOWN，不能宣稱 UNSAT。

## 4. 預設銜接及回退

`solver/ternary_signs_native.py` 與 `solver/ternary_system_native.py`：
若 constant-weight conditioning 可用，仍使用該精確分割；
否則 d=43、自由變數 16..24 時自動使用模 8 kernel。
`quadratic=False` 可回退原 raw 枚舉。其他長度保留原自動選擇。
這已接入原 support completion，使用 main 的既有執行方式即可套用。

`solver/compressed_lift.py`：L86 且指定 parity content 時，預設先用
`solver/odd_lift_native.py` 的完整 alternated-difference sign completion。
其公式與原始 binary 枚舉驗證見
[差值分解文件](L86_weight_conditioning_and_joint_search.md)。
SAT 必須通過獨立 full verifier；完整 UNSAT 只排除這個固定 job。
UNKNOWN 回原 Z3 路徑。含 blocked orientations 或 cross constraints 的查詢保留一般 Z3。
`native_odd_lift=False` 回退此前 GF(2)/Z3 方法。
其最多 2^23 assignments 的前處理不計入 Z3.check 的 timeout_ms。

## 5. 驗證與計時的範圍

`tests/test_ternary_quadratic_signs.py` 比較 d=3、5、7、9、11 的全部自由變數配置，
二次函數與直接整數 PACF／計數的第二 bit 必須逐一相同。
精確 SAT、UNSAT 及截斷 UNKNOWN 亦與原 raw kernel 和 literal 可行集合比較。
另用 L43、21 free bits 的高位配置及 100 個固定 seed 配置驗證係數，
並完整枚舉已找到的 L86 coarse 候選之差值 supports。

第一輪相同 256 supports、五次交錯重播，包含 native affine-plan construction：
raw 中位時間 4.7302 秒，單獨模 8 為 3.1393 秒，減少 33.63%。
每輪 assignments 都是 536,870,912，結果完全一致。
完整 correlation checks 從 34,614,520 降到 136。
記錄位於 `results/L86_mod8_signs_20261007/report.json`。
背景 sparse 長測同時執行，交錯比較可減少偏差但不能消除硬體負載變動。

第二輪比較四種模式，仍為相同 256 supports、五次交錯重播：
raw 4.4601 秒、只有模 8 為 2.9902 秒、只有 constant-weight 為 3.7736 秒，
自動組合兩者為 2.5522 秒。組合方法比 raw 少 42.78%，
比此前 constant-weight 模式少 32.37%。所有模式的完整判定仍一致。
記錄：`results/L86_mod8_hybrid_20261007/report.json`。

最後使用現有 production `complete_support_signs` 介面再比較一次，
兩側都啟用原本 constant-weight 與內容剪枝，只切換模 8 filter。
相同 256 supports、五次交錯重播，包含原 Python GF(2) 前處理：
此前模式中位時間 4.4648 秒，新模式 3.2095 秒，減少 28.12%。
每輪都枚舉 380,257,664 個 assignments，完整判定一致。
這個比較最接近本次 production 符號補全的實際變更範圍。
記錄：`results/L86_mod8_production_api_20261007/report.json`。

真實 MPS 驗證：32 lanes、64 步、49 次跨內容提案獲接受。
全部 64 個 trace frames 逐一重算 Python 整數 PACF，
並核對兩列 parity content 可行性及每列 support 至少 7；全部通過。
記錄：`results/L86_mod8_mps_smoke_20261007/report.json`。
這項 MPS 驗證針對 GPU 內容剪枝／transfer 約束，不是 native CPU kernel 的效能計時。

回歸測試：1,266 passed、14 skipped；既有兩個環境相關 benchmark test modules
`test_mps_acceleration_benchmark.py`、`test_torch_mathematical_benchmark.py` 未納入此命令。
主程式提示及 staged native metadata 修改後，另有 31 個對應測試通過。

這是固定支持符號補全的收益，尚未證明「找到全新 L86 PQCP」快 20%。
必須以完整 verifier、等價類去重及相同搜尋預算的實際 discovery rate 驗證後者。

## 6. 新增尾端商空間投影：先證明整個 prefix 不可行

將自由變數 f=(p,t)，t 為小尾端，將所有尾端二次係數 q_ij 的 span 記作 H。
以 GF(2) 線性商映射 π 消去 H，則對固定 prefix p：

\[
\pi Q(p,t)=\pi Q(p,0)+\sum_j\pi L_j(p)t_j.
\]

右側對 t 是線性方程。若其不相容，該 prefix 的全部 2^|t| 配置都可嚴格排除。
若相容，Gaussian elimination 取得 particular solution 與完整 nullspace，
枚舉它們產生的每個 t，再檢查原 Q、精確負號數及整數 PACF。
dependent columns 保留對應 nullspace，不能只留下單一解。
因此這是必要條件的精確加速，不依賴 generic MQ 或 semi-regularity 假設。

d=43、16..24 free bits 預設保留 6 個 tail bits；其他情況回原 kernel。
constant-weight 分割後亦可在每個 fibre 套用此方法。
budget 不足時維持原有 bounded ordering／UNKNOWN 語義。
`quadratic=False` 回原 raw kernel；`hybrid_tail=0` 回逐點 Gray kernel；
`conditioned_hybrid=False` 回此前 constant-weight 枚舉。

注意新 `assignments` 表示已覆蓋、包含整個 prefix 排除在內的配置數，
不再表示真的逐點訪問次數。另記 `prefix_branches`、`projected_candidates` 呈現實際工作量。
不能拿此統計與舊逐點計數直接推算吞吐率。

相同 256 supports、五次交錯重播：Gray 中位 3.2269 秒、tail=6 為 1.4737 秒，
階段時間減少 54.33%；完整判定、模 8 通過數、精確檢查數相同。
記錄 `results/L86_mod8_quotient_20261007/report.json`。
production `complete_support_signs` 比較此前 Gray 2.9256 秒與 quotient 1.8507 秒，
額外減少 36.74%，每輪覆蓋 380,257,664 配置、判定相同。
記錄 `results/L86_mod8_quotient_production_20261007/report.json`。

從兩個 sparse coarse populations 各均勻 reservoir 抽樣 512 literal pairs，
共 1,024 組、五次配對重播：原 conditioned 模式 1.40524 秒，新模式 0.42453 秒，
減少 69.79%，全部 UNSAT、覆蓋 479,363,072 配置。
這些案例只比較符號求解階段；不得當成新 PQCP discovery 加速。
記錄 `results/L86_conditioned_projection_20261007/report.json`。

## 7. 精確 projected-cardinality 可行性

指定某列恰有 n 個負號時，其 projected affine checks 為 Hs=b。
當 binomial(w,n)<=4096，完整枚舉這些 constant-weight 模式並計算通過 checks 的數量。
若為零，整個符號問題 UNSAT，無須枚舉其餘列；若非零，用實際相容數估算 fibre 成本。
超過上限保留原選擇，不把未檢查模式當成不可行。
另先使用 fixed_negative<=n<=fixed_negative+flexible 的必要上下界。
`cardinality_pruning=False` 可回退舊預檢查與成本選擇。

weight 7 negative 的最後 UNKNOWN 就由此條件直接排除：
第一列雖然有一個 fixed negative 與五個 flexible bits，單純上下界仍不能排除 n=1；
但 projected image 的四個配置只容許負號數 3 或 5，故所有七個 n=1 模式皆不相容。
這避免把原 2^24 的補全工作誤當成未知或必須逐點檢查。

## 8. 最終流程驗證

新的內容下界為 8，完整範圍與回退條件見 [內容剪枝](L86_sparse_content_pruning.md)。
候選 formation 亦已接入 [最小 Hamming 半週期補充](L86_half_shift_observation.md)，
保留原 quantization、loss 與 Adam 更新。
本次最終完整回歸為 1300 passed、14 skipped，另有兩個 boundary expansion tests 通過；
兩個既有環境 benchmark modules 的排除仍明確保留。
真實 MPS 最終記錄 `results/L86_minimum8_energy_mps_verification_20261007/report.json`。
尚未找到完整 L86 新解，不能以本文件任何階段速度數字宣布 discovery 目標完成。
