# 任意位置、跨內容的精確翻轉配對

這是本專案自行推導的模 8 必要條件與原始二元空間的精確局部搜尋。
不同於舊的固定 parity counts 交換修復，也不同於連續窗口修復，
本方法允許每列任意位置至多 r 次翻轉，包含奇數次翻轉及內容變更。
目前沒有完整 L86 新解，不能宣稱新解發現速度提升 20%。

## 1. 原始相關值的二次整數展開

設位元 b_i、符號 s_i=1-2b_i，翻轉位置集合 F，座標均模 L。
對任意非零位移 u，直接展開得到

\[
\frac{P_{s'}(u)-P_s(u)}4=
-\frac12\sum_{i\in F}s_i(s_{i+u}+s_{i-u})
+\sum_{i<j,\ i,j\in F}s_i s_j
\left(\mathbf1_{j-i\equiv u}+\mathbf1_{j-i\equiv -u}\right).
\]

第一項為整數，因為括號是 -2、0 或 2。
第二項保留兩個方向；u=L/2 時必須計兩次，不能漏掉 midpoint。
將此精確整數式模 2 約化，單點翻轉的第 u 座標為

\[
\sigma_i(u)=1\oplus b_{i+u}\oplus b_{i-u}.
\]

任兩個翻轉位置額外在其無向距離座標 XOR 1；
midpoint 的二次項為 2，模 2 則為 0。
所以一列任意 r 次翻轉的完整模 8 差分，可用 r 個單點簽章及
最多 r(r-1)/2 個位移位元精確算出。這沒有忽略交叉項或近似相關值。

## 2. 必要配對與不遺漏 preimages

L 偶數時原始兩列相關和 T(u) 是 4 的倍數。
令 c(u)=T(u)/4 mod 2，獨立位移 u=1..L/2 打包成整數。
任何嚴格 PQCP 修復都必須滿足

\[
\sigma(F_A)\oplus\sigma(F_B)=c\oplus e_k,
\quad1\le k<L/2.
\]

±4 在模 8 等價，故此階段合併正負號；最後用整數相關恢復 η。
半週期座標始終要求為零。
索引列 A 的所有翻轉集合，再逐一列舉列 B，查找所有可能 k。
簽章相同的 literal preimages 全部保留，不使用 Bloom filter、
homometric 代表、浮點或雜湊碰撞作排除。
每個簽章匹配先檢查 DC 與 alternating sum，接著重算所有獨立 PACF。
Python 再獨立驗證完整原始長度、目標 profile 與兩列 Hamming 距離。

L86 可另用逐列必要條件：
ΣA²+ΣB²=164 或 180，所有偶整數平方分解分別為 8²+10²、6²+12²。
alternating sum 也滿足同樣兩個總平方值之一。
因此單列 DC 及 alternating sum 的絕對值都必須屬於 {6,8,10,12}。
這個刪減只在 L86 使用；其他長度不套用這些數字。

### 將 DC／ALT 必要條件提前到索引

把 {6,8,10,12} 編成 codes 0,1,2,3，另一列必須取 code 3-code。
DC pair 位於 {6,12} 表示 η=+1，位於 {8,10} 表示 η=-1；
ALT pair 決定 η(-1)^κ。
兩者符號相同時 κ 必為偶數，不同時必為奇數。
因此只需查詢 21 個 κ，把另一列的 DC/ALT absolute codes 直接併入精確索引 key。
符號正負全部保留，這不要求固定 parity contents 或 row sums 的方向。
`content_grouping=False` 為可回退的完整原查詢方式。
兩側都使用精確遞增 DC/ALT sums，配對比較只切換這個數學剪枝。

六個 GS 中心、三次交錯順序重播、共 18 個固定工作量配對已完成。
每個 product ball 的覆蓋數、索引候選數與實際完整 PACF 檢查完全相同；
新方法的 κ 查詢數恰為一半。
呼叫時間中位數從 1.0187988335 秒降到 0.5445109585 秒，减少 46.5536%。
記錄 `results/L86_unrestricted_grouping_paired_20261007/report.json`。
**兩側完整新解皆為零；這是局部固定工作量時間，不是新解發現速度。**

## 3. 完整範圍、成本與限制

每列枚舉数為 N_r=Σ_{j=0}^r binom(86,j)，涵蓋原中心及全部較小半徑。

| 每列半徑 | 每列集合數 | 原始兩列乘積 | 24-byte 索引配置上界 |
|---|---:|---:|---:|
| 4 | 2,229,637 | 4,971,281,151,769 | 53,511,288 bytes |
| 5 | 37,055,939 | 1,373,142,615,171,721 | 889,342,536 bytes |

這是兩個每列半徑球的乘積，不是總距離 ≤2r 的完整球，
例如 r=4 不包含同一列翻轉 8 次、另一列不變的配置。
UNSAT 只排除當前中心的上述乘積鄰域，不能限制其他中心或一般 L86 搜尋。
SAT 回傳第一個完整驗證解，並明示不是全部 SAT models 的枚舉。
記憶體配置、編譯或原始整數核對失敗直接報錯，不能當作 UNSAT。

`solver/unrestricted_flip_join.py` 與 `csrc/unrestricted_flip_join.c` 為核心；
`experiments/repair_unrestricted_flips.py` 可在 IDE 按執行，
預設讀取根目錄 logs 中修改時間最新的 L86 torch JSONL，重新核對每列 literal profile，
選取最多 16 個非解中心，先涵蓋各 observed parity/sign family，再以精確能量排序。
不信任 log 中的 verified 或 target_energy 欄位，實際計算原始相關及能量。
沒有 log 才改用 GS 原料；選中心是排程策略，不是一般搜尋的數學排除。
以 `DEFAULT_RADIUS` 調整 4 或 5，不必輸入指令。
每次建立獨立時間戳結果目錄、來源快照及原料副本。
主程式預設尚未加入，避免局部修復成本擠壓既有完整搜尋的時間。

## 4. 已完成驗證

專項測試以 L6 所有非解中心及其全部每列 radius-1 鄰域比對獨立 literal oracle，
同時包含 SAT、UNSAT 與內容改變正例。
另對 6/14/68/86/94 點直接比對上述二次式和原始整數 PACF，包含 midpoint。
已知 L68 解只用於跨第 64 位元的 kernel 正例控制，不作為發現實驗種子。
另檢查半徑 4、5 的小長度完整列舉數與 L86 DC 必要條件排除。

六個 GS 中心的 radius-4 乘積鄰域均完整 UNSAT：
42 個模 8 必要匹配，11 次完整 PACF 座標檢查；
記錄 `results/L86_GS_unrestricted_radius4_20261007/report.json`。
已完成 direct-Metal chains 的 60 個 best records 去重為 10 个中心，
radius-4 均完整 UNSAT，共 72 個模 8 匹配、23 次座標檢查；
記錄 `results/L86_joint_unrestricted_radius4_20261007/report.json`。
這是完整有限鄰域結果，不是只測幾個隨機翻轉，也不是新解速度證據。
同樣十個中心的 radius-5 已完成，全部 UNSAT；
11,180 個模 8 必要匹配、4,665 次完整 PACF 座標檢查，elapsed=402.787094166 秒。
記錄 `results/L86_joint_unrestricted_radius5_20261007/report.json`。
每個中心的 37,055,939 個逐列集合均完整覆蓋，沒有 UNKNOWN 或中斷。

根目錄最新 `logs/torch_L86_seed123.jsonl` 的 16 個 independently audited centers
也已完成 radius-4，全部 UNSAT；
記錄 `results/L86_latest_log_unrestricted_radius4_20261007/report.json`。
這次輸入為實際長測候選，未用既有完整解初始化。
共 114 個模 8 必要配對、39 次完整 PACF 座標檢查，elapsed=22.124597 秒。

本輪整合專項驗證為 26 passed，涵蓋新翻轉配對、仿射格、舊整數核、
邊界投影及原固定內容 4+4 修復；不是整個專案的最新全面回歸。

後續又完成最新長測十六個中心的 radius-5 product balls，全部 UNSAT，
elapsed=370.907356042 秒；
記錄 `results/L86_latest_log_unrestricted_radius5_grouped_20261007/report.json`。
八個新 D-optimal 原料的最佳 interleave seeds 也完成 radius-5，全部 UNSAT，
記錄 `results/L86_new_doptimal_unrestricted_radius5_20261007/report.json`。

目前邏輯正確與局部空間完整性已驗證，但仍需用新鮮的一般搜尋候選，
比較加入修復前後的完整新等價類發現時間，才能決定是否接入主程式預設。
