# L86 稀疏內容剪枝的完整證明與驗證

2026-10-07。這裡排除的是不可能達到精確目標的整個內容類別，
不是因為短測沒找到解就把候選關掉。
目前仍未找到完整的新 L86 解，不能宣稱已達成 20% discovery 加速。

## 解析下界：每側至少 5 個非零位置

設 normalized ternary pair a,b 長度 d=43，總非零個數為 43。
令 U,V 為兩側 support，W=1−V。若 |U|=w，則 |W|=w。
在奇數週期長度下，非零位移有

P_(1−W)(u)=d−2w+P_W(u)。

目標 P_a+P_b 在獨立位移 k 為 η∈{−1,+1}，其他 20 個獨立位移為 0。
取 mod 2 得

sig(U) XOR sig(W)=所有 21 位均為 1，再 XOR 第 k 位。

因此 P_U+P_W 在 20 個位移必須為奇數，各至少是 1。
這些 unsigned 自相關在全部獨立位移的總和是

2×binomial(w,2)=w(w−1)。

所以 w(w−1)≥20，推出 w≥5。交換兩側可知两側都至少有 5 個非零位置。
一般奇數半長度的解析版本是 w(w−1)≥d//2−1，
已實作於 `solver/support_size_bound.py`，以每個小長度的 literal parity supports 驗證。

## 有限完整枚舉：將下界提高至 7

普通 Fourier character 给出

(Σa)²+(Σb)²=43+2η。

η=+1 時，兩側 row-sum 絕對值只能是 6、3；η=−1 時只能是 5、4。
由 w≥|Σa|、w≡Σa (mod 2)，小側 w=5 或 6 只需處理四個內容形態：

| 稀疏 weight | η | 兩側 negative counts | 完整 parity 匹配的 ordered 支持組合 |
|---:|---:|---:|---:|
| 5 | −1 | (0,17) | 2 |
| 5 | +1 | (16,1)，第一側為 complement support | 2 |
| 6 | +1 | (0,17) | 3,438 |
| 6 | −1 | (16,1)，第一側為 complement support | 3,438 |

負 row sum 可整側換號，row roles 可交換；這些操作不改變 periodic PACF。
43 為質數，共同 unit decimation 可以把任意非零 normalized target shift 轉成 2。
因此四個形態覆蓋全部符號、角色及非零位移。

對 U,W 各枚舉全部固定 weight 的 binary bracelets：
即將循環位移與反轉等價的 supports 保留一個代表。
兩側可獨立旋轉與反轉，因為每側的 periodic PACF 不變；
符號求解會枚舉 supports 上的全部指定個數符號。
相同 parity signature 的所有不同 bracelet preimages 均保留，沒有把 homometric supports 合併。

43 是奇質數，非平凡固定 weight word 的 rotation orbit 均為 43。
Burnside 計數給出

bracelets(43,w)=[binomial(43,w)/43+binomial(21,floor(w/2))]/2。

weight=5 有 11,298 個，weight=6 有 71,554 個。
C 生成器的輸出數與此精確公式相同；
對 d=3、5、7、9、11 的每個 weight，亦與全部 literal masks 的 dihedral orbits 比較。

完整 parity signature matching 將四個形態縮到表中 6,880 個支持組合。
每個組合先建立必要 mod 4 GF(2) 方程；相容者再完成全部指定符號個數的 affine 枚舉。
共實際枚舉 1,578,762,240 個 assignments，全部是完整 coarse UNSAT。
此處沒有使用 lift 的失敗代替 coarse UNSAT：證明的對象是 coarse 必要條件本身。

因此可得到本專案的有限計算證明：

**所有符合精確 L86 folded target 的 normalized ternary pair，兩側非零位置皆至少為 7。**

`solver/l86_content_certificate.py` 每次程式啟動的第一次需要時重新執行證明，
並檢查完整 bracelet 數量；本機首次約 7.3 秒，之後只取 bounded cache。
任何 SAT、UNKNOWN 或生成容量不足，都只保留解析下界 5，不能使用下界 7。
它不是未核對就相信舊 JSON 的静態排除清單。

## 預設搜尋如何使用

`paired_contents` 仍回傳全部半位移與 parity content 可行的 contents。
新 `search_paired_contents` 只在 L86 搜尋初始化時使用已驗證下界，
排除不可能精確匹配的內容類別；其他長度保留原方式。
對四個 canonical profiles，原本 17、17、18、18 個 contents。
只驗證 weight 5/6 時各剩下 15 個；下方新增 weight 7 完整證明後各剩下 14 個。

`complete_support_signs` 遇到非零個數少於已驗證下界的 L86 reference，
可以直接回報該固定 support 的 UNSAT，而不再花時間重新枚舉。
`content_pruning=False` 保留原方式供對照。
GPU 的跨內容 transfer 亦使用同一必要範圍：
ceil((support_lower−σ)/2)≤n≤floor((d−support_lower−σ)/2)。
允許的 n 範圍仍是連續區間；各內容內的 symbol swaps 可到達全部排列，
相鄰內容可透過原本的 opposite transfers 連接。
所以去除沒有解的端點內容不會去除可行解或切斷這個合法 proposal graph。
native 延續仍可探索其他中間態；其後精確補全會套用同一已驗證排除。

L86 的直接三值壓縮搜尋預設使用此剪枝，IDE 執行 main.py、
選擇啟用直接三值壓縮時不需要增加指令。
cycle snapshot 記錄 content_support_minimum、content_certificate_verified、
content_certificate_cases，方便核對實際使用的證明。

只使用最初下界 7 時，原 70 個初始化內容選項中保留 60 個。
目前完整驗證下界 8 時，coarse 篩選保留 56 個，再加 binary 證明保留 53 個。
這只描述候選內容的必要條件，不是已測得的新 PQCP discovery 加速百分比。

## 下一層的完整搜尋

同一 bracelet/signature 方法也能搜尋較大 sparse supports：
固定 n_A 後，|U|=σ_A+2n_A，W=1−V 仍與 U 同大小。
若兩者太大，可以一起取 complement；兩個 parity signatures 的 XOR 不變。
再以完整 `complete_pqcp_supports` 求解 coarse 與差值兩個獨立問題。

η=−1、n_A=1（sparse weight 7）有 375,364 個 bracelets，
130,142 個完整 parity 匹配的 ordered 支持組合。
even target k=2 與 odd target k=41 各已完整測試，均沒有完整 L86 解。
它們分別代表原始 even/odd shift 的兩個 unit-decimation 類。
這是這兩個固定內容／target 類的 full UNSAT，不是 coarse UNSAT：
even 有 294 組、odd 有 228 組由差值問題排除。
這個統計本身不表示 coarse 問題必定有解；求解器也會先檢查兩側線性矛盾。
coarse witness 是否存在須由獨立 coarse 求解結果確認。

記錄：`results/L86_sparse_n1_negative_20261007/`、
`results/L86_sparse_n1_negative_odd_20261007/`。
η=+1、n_A=1 的 even target k=2：2,716,258 個組合已全部完成，
均為 full UNSAT，其中 11,068 個由差值問題排除，其餘由 coarse 問題排除。
記錄：`results/L86_sparse_n1_positive_20261007/`。
相同內容的 odd target k=41 已完整完成 2,716,258 組，全部為 full UNSAT，
其中 2,706,473 組由 coarse 問題排除，9,785 組由差值問題排除。
記錄：`results/L86_sparse_n1_positive_odd_native_20261007/`。
四個 n_A=1 代表類別共 5,692,800 組，均完整排除。
這些早期紀錄沒有保存 coarse witnesses；新版搜尋腳本會另外保留見證及 UNKNOWN，
仍以 aggregate coverage 記錄完整範圍。

## 完整二元內容剪枝

`solver/l86_binary_content.py` 使用上述四個已完成的完整枚舉結果。
它檢查固定記錄的 SHA-256、精確 target、bracelet 數、全配對覆蓋、
沒有 UNKNOWN/SAT、兩種排除數相加吻合等條件。
SHA-256 是記錄的身分及完整性檢查，**不是獨立 UNSAT 證書驗證器**。
數學依據是完整 bracelet 生成、完整兩側 sign enumeration 的演算法正確性，
以及確實完成的有限計算；不能把一般長測沒找到解當成這個證明。
缺少或修改記錄、scope 不符，會停用此較強剪枝。

對奇數原始位移，令 x'_i=(−1)^i x_i，y 同理。
則 T'(u)=(−1)^u T(u)，η 翻轉；even ones 不變，odd ones 變為 43−o。
對 compression/difference 分解，c'=g、g'=c，故 n'_A=e_A−n_A。
兩個 canonical odd profiles 都有 e_A=16，且互相對應。
因此 n_A=1 的完整排除亦證明另一符號 profile 的 n_A=15 完整排除。
even k 的 modulation 不翻 η，與 row swap 合併後 n_A 不變，沒有據此排除 even n_A=15。

當獨立的 coarse support>=7 證明也已驗證時，原允許 n_A=1..15。
canonical odd targets 現為 2..14，even targets 為 2..15；n_B=17−n_A。
其他 numeric content profiles 保持原處理，避免套用未驗證的 canonical 假設。
只使用 coarse 下界 7 時，四個 raw 類別共 70 個內容選項，coarse 剪枝後 60 個，
完整二元剪枝後 54 個。最新下界 8 的對應數量見下一節。
這是排除內容選項的數量，不能當作新解搜尋時間的速度比例。

GPU 啟動抽樣及跨內容提案均使用這些連續區間，native continuation
在啟動內容位於區間內時也保持區間。各內容的 symbol swaps 仍可遍歷其排列，
opposite transfers 連接相鄰 n_A，區間內的 proposal graph 保持連通。
完整 fixed-support completion 與 binary lift 也可直接排除證明涵蓋的內容。
**coarse-only sign solver 不套用此 binary exclusion**。
`binary_content_pruning=False` 可回退；有限枚舉重現腳本明確關閉此選項，避免循環證明。

## 新增 weight 7 coarse 完整排除：下界提高至 8

ordinary character 與全列換號／row swap 將 sparse weight 7 化成兩種形態：
η=−1 的 (n_A,n_B)=(1,16)，η=+1 的 (15,2)。後者第一側為 dense support。
各有 375,364 個 weight 7 bracelets、130,142 個完整 parity matching ordered pairs。
這次求解只檢查 coarse C，沒有用 G 失敗替代 coarse 排除，也沒有套用既有內容排除。

positive 的 130,142 組全部完整 coarse UNSAT。
negative 的第一輪是 130,141 組完整 UNSAT 與一組預算不足的 UNKNOWN；原始記錄保留。
例外 masks=(67375149,8795518140394)，負號數 (1,16)，原 affine space 有 24 free bits。
再完整覆蓋 2^24 配置後確認 UNSAT；亦有更小的獨立證明：
投影到第一列的 affine image 只有四種位元配置，其負號數只能是 3 或 5，
指定負號數卻是 1。逐一檢查七個單負號模式全部違反 projected GF(2) checks。
Python 獨立重建此投影與七個模式的失敗，不依賴 C kernel 的回報。

完成記錄為 `results/L86_sparse_coarse_n1_negative_verified_20261007/report_completed.json`
及 `results/L86_sparse_coarse_n15_positive_20261007/report.json`。
第一份 completed record 明確保存原 UNKNOWN 與 refinement；沒有修改原始失敗歷史。
runtime 檢查兩份固定 SHA-256、target scope、完整生成／匹配數、無 UNKNOWN／SAT，
並重算例外的 Python 證明。這是對實際完成的有限計算記錄的認證，
**不是獨立的通用 UNSAT 證書檢查器**。缺檔或檢查失敗會回退下界 7，
連 weight 5/6 的 runtime 證明也失敗時只使用解析下界 5。

因此 coarse 與 complementary G 的兩側非零個數皆至少 8。
總覆蓋內容案例數為 6,880+2×130,142=267,164。
四個 canonical profiles 的 coarse 內容各剩 14 個，共 56 個；
加上完整 binary 的端點排除，共剩 53 個：odd 兩類各 13、even negative 14、even positive 13。
GPU 初始抽樣及跨內容 transfer 自動使用相同已驗證連續範圍。

真實 MPS：32 lanes、64 步、43 次 transfer 獲接受。
64 個 trace frames 全部以 Python 整數 PACF 重算，內容範圍與 support>=8 均通過；
亦執行既有 direct-compression round 與預設 cyclic full-support 銜接。
記錄 `results/L86_minimum8_mps_verification_20261007/report.json`。
這是正確性驗證，沒有量測新 L86 解的 discovery rate。
