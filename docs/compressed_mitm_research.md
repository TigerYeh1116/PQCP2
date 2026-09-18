# 受限制壓縮枚舉、精確 MITM 與分層展開：可行性研究

研究日期：2026-09-19。這是獨立研究，不是正式 backend 的變更。
沒有讀取既有 PQCP／搜尋日誌來產生候選，沒有寫入 L.txt，也沒有修改
solver、main.py、數學定義或正式參數。研究程式的 stdout 是 JSON。

## 結論

1. L=44 的「短空間枚舉＋整數 PACF 配對」已實際完成一個 canonical
   target/content 案例：235,325 個 cyclic representatives，16,928 個匹配，
   此次約 3.93 秒。這些是壓縮候選，不是已驗證 PQCP。
2. L=68 的初步嚴格 content／norm／margin 限制後，每側仍有約
   6.21 億至 12.20 億個 cyclic representatives（依 target/content 而異）。
   尚未逐序列做 PSD 篩選，不能把這個數量稱為最終候選數。
3. 值得優先研究「固定粗模式後，精確 MITM 列舉中間層」，減少 Z3
   同時處理中間模式與完整序列的負擔。第一個 L44 案例可在約 0.28 秒
   列出全部 22 個中間配對，但尚未證明整體求新解的速度提升。

## 數學來源與可借用的方法

專案定義來源：`文獻/06_PACP.pdf` Project 2 第三題，以及現有
`correlation.py`、`verifier.py`。非零 shifts 共兩個，均為 magnitude 4。
對偶數 L，對稱性使目標為 T(0)=2L、T(k)=T(L-k)=4 eta，其他為零，
其中 0<k<L/2、eta=±1。

文獻：Lumsden、Kotsireas、Bright，*New Results on Periodic Golay Pairs*，
2025 修訂版，https://arxiv.org/html/2408.15611v3 ，特別是第 3、4 節。
可借用 fixed-content/orderly generation、signature matching 與 multilevel
uncompression 架構；文獻的零 sidelobe 目標不能直接套到 PQCP。
本研究使用完整整數 PACF signature，不使用四捨五入的浮點 PSD 作配對 key。

## 一、完整壓縮恆等式

L=m*d，x 為 ±1 序列：

    c_m(r) = sum_{j=0}^{m-1} x(r+j*d)
    P(c_m;r) = sum_j P(x;r+j*d)

故合法 pair 必須滿足 P(c_a)+P(c_b)=fold_m(T)。推導只需展開乘積，
依原始兩索引之差 modulo L 分組。這是必要條件，不是充分條件。
所有 target aliasing 都要保留；特別是 k=d 時，兩個 sidelobe 都落到
短 profile 的零位移，不能錯用固定 2L 的短序列 norm。

## 二、從 content 開始，而不是枚舉全部 5^n

L=4n（此處 n=11 或17），令 q=c_4/2，則 q∈{-2,-1,0,1,2}。
設各字母數量為 h_-2,...,h_2，必須滿足：

    sum h_j = n
    sum j*h_j = (L-2*w)/2
    Q = sum j^2*h_j
    Q_A + Q_B = fold_4(T)[0]/4

其中 w 由現有普通總和／alternating sum 必要條件選出，不猜 weight。
L68 的最後一式通常為34；k=17、eta=-1時為32，eta=+1時為36。

研究程式完整枚舉 histogram，再用 multinomial 計數，不枚舉數億條序列。
固定 content 的有標記字串數為 n!/product(h_j!)。因11與17為質數，
非 constant 字串的 cyclic orbit 長度必為 n，所以其 necklace 數精確為
multinomial/n；constant case 單獨記為1。不是粗略除以n的估算。

## 三、雙向壓縮的嚴格相容性

因 gcd(4,n)=1，位置 i 可寫成 (i mod4,i modn)。factor-4 壓縮給 n 個
column sums，factor-n 壓縮給4個 row sums。對應的 binary degrees：

    column_degree_j = 2-q_j
    row_degree_i = (n-r_i)/2

先完整枚舉4個 row sums，同時要求：普通／奇偶 weight、完整
factor-n folded PACF 目標。再用 Gale--Ryser 的等總和及四個 majorization
不等式，精確判斷兩組 margin 是否存在共同 binary matrix。

每個 A histogram 僅在存在某個 B histogram、兩者 norm 互補、且存在
同一個合法 row-pair 使雙方 margin 可行時保留。這仍然不保證完整 PACF。
Histogram 計數可用排序的 column degrees，但生成序列時不能任意重排
column 當成 PACF 等價。

本研究所發現的「完全無 row-pair」profile 與既有 quarter-frequency
pruning 一致；程式逐一 assert，不把已存在的剪枝宣稱成新發現。

## 四、精確空間計數

以下每列固定 canonical target/content；內容為 (A_even,A_odd,B_even,B_odd)。
只做 cyclic reduction，尚未反轉、共同 decimation、完整 job canonicalization、
逐序列 PSD 或完整短 PACF 配對。各列有重疊，不能相加當作不等價總數。

| L | k | eta | content | A necklaces | B necklaces |
|---|---|---|---|---:|---:|
|44|2|-1|(8,10,8,12)|119,132|116,193|
|44|4|-1|(7,11,9,11)|119,132|116,193|
|44|4|-1|(8,10,8,12)|119,132|116,193|
|68|1|-1|(12,18,15,15)|864,445,100|864,445,100|
|68|1|+1|(12,16,15,19)|1,021,404,082|831,120,403|
|68|2|-1|(13,17,13,17)|864,445,100|864,445,100|
|68|2|+1|(14,14,14,20)|1,021,404,082|831,120,403|
|68|4|+1|(11,17,17,17)|1,021,404,082|831,120,403|
|68|4|+1|(14,14,14,20)|1,021,404,082|831,120,403|
|68|17|-1|(12,18,15,15)|620,505,404|620,505,404|
|68|17|+1|(12,16,15,19)|1,220,134,042|1,141,558,531|

排除的 canonical profiles：L44 (k2,eta-1,content7,11,9,11)；
L68 (k2,eta+1,content11,17,17,17) 及 (k4,eta-1,content13,17,13,17)。
這是指定 target/content 不可能，不是宣稱整個長度無解。

對 L68，僅620,505,404筆、每筆假設24 bytes packed payload，已約14.9 GB
（十進位），尚未算 hash overhead。Python tuples/dicts 會更大。
因此不能把全部候選直接塞進單一記憶體 hash table。需要精確 PSD
prefix pruning、按 norm/content 分區，必要時外部排序／分桶。

## 五、小型全枚舉正確性

全部枚舉 L=4、8、12 的 binary sequences，透過 baseline correlation
建立獨立完整 PACF reference；沒有讀入任何已找到的 PQCP。
同時獨立枚舉所有五字母粗模式及所有 ternary 中間模式。

|L|完整 binary 單條數|粗模式配對|可展開到中間層|有完整 binary 解的粗配對|
|---|---:|---:|---:|---:|
|4|16|5|5|5|
|8|256|16|16|16|
|12|4,096|936|936|816|

所有配對數包含 ordered A/B、標記座標及各 (k,eta) 案例，不是去等價解數。
L12 的 cross-compression 檢查將936降到864，仍包含全部816個真正可展開
配對，另有48個 false positive。中間層條件在這組小型資料中沒有再減少
粗配對數，不能假定每層剪枝都必然有效。

每個 binary word 都檢查壓縮 PACF aliasing 和 factor4→factor2 的組合關係；
每個可行粗配對都有獨立 verifier 核對 witness。研究腳本 assert：

    binary_feasible ⊆ middle_feasible ⊆ coarse_matched
    binary_feasible ⊆ cross_compatible

## 六、L44 精確 MITM pilot

固定 k=2、eta=-1、content=(8,10,8,12)。FKM 枚舉各 histogram 的全部
cyclic representatives，B建表，A串流查表；每個 signature 保留所有候選。
長度11的整數 PACF 只需保存 u=0..5，其餘由 periodic symmetry 決定。

- A：119,132；B：116,193，與純組合計數完全相等。
- B signature buckets：43,842；最大 bucket：14。
- 完整 signature 配對：16,928；本案例 cross check 後仍16,928。
- 建表1.978秒；A生成及查表1.948秒，合計約3.93秒（單次量測）。
- 未使用既有 PQCP、GPU、浮點篩選或正式搜尋參數調整。
- 只去 cyclic 重複，16,928不代表完整對稱性下不等價的工作數。

依枚舉順序取前4個配對，現有 multilevel Z3 每個設250ms：皆UNKNOWN，
都尚未找到中間 model。這是小 budget 的診斷，不能據此判定不可展開。
check timeout 不含所有 Python建模時間，實際4次約1.14秒。

## 七、替代實驗：固定粗配對後，中間層也做 MITM

對第一個粗配對，直接枚舉每個位置所有 (x,c-x)，x,c-x∈{-2,0,2}，
放到 i 與 i+n。raw count 精確為：

    3^(粗模式零數) * 2^(粗模式 ±2 數)

先加中間層 even/odd sum，然後按 fold_2(T) 做完整整數 PACF 配對。

- raw A/B：34,992 / 576。
- parity content 後：5,296 / 70。
- 全部精確中間配對：22；cross 後仍22。
- 枚舉及配對約0.277秒。
- 前4個交給現有 binary Z3（每次check 250ms）：2 UNSAT、2 UNKNOWN。
- 其餘18個沒有測；此粗工作仍然 unresolved，不能宣稱 UNSAT。
- 本次沒有找到完整 PQCP，也沒有證明新解速率提升。

前4個是 deterministic enumeration-order 樣本，不是代表性成功率調查。
這個試驗的意義是：可將中間層的22個候選明確交給 binary completion，
不必讓 Z3 先在較寬的中間層編碼中尋找它們。是否普遍划算仍須測試不同
content、粗模式與L68；不能把一次0.277秒推廣到所有工作。

## 八、建議下一步與完整性邊界

1. 先建立獨立 L44 研究 backend：fixed-content FKM→精確 hash join→
   壓縮 job canonicalization→分層展開；不立刻替換正式 GPU/SA 流程。
2. 中間層先計算兩側 raw split 數；小工作用 MITM，大工作保留現有 Z3／
   staged completion。這是工作排程，不是把大工作剪掉的數學規則。
3. L68先研究 C 的 prefix generation、可認證 PSD bounds、分區／外部
   sort-merge。現有 `certified_psd_pruning_from_profiles` 可直接接整數
   壓縮 PACF 與 folded target，但成本要實測，不能用不可靠 float 拒絕。
4. 固定剩餘content的 prefix bounds 必須包含所有未指派排列；若做
   partial PACF bounds，未決交叉項不可直接設零。
5. 共同 decimation 不能對A/B各自獨立 canonicalize；獨立 rotation 可行，
   但須保留正確 parity/row metadata。本文 necklace reduction 可由原序列
   旋轉4的倍數實現：gcd(4,n)=1，且保持所有 mod4 row counts。
6. 所有 cap/UNKNOWN 必須可回訪，只有完整展開全部UNSAT才能關閉父工作。
   成功解仍用獨立 verifier，再走既有完整等價去重與 L.txt writer。

最終評估指標仍是「固定資源找到的不等價新 PQCP 數／時間」，不是只有
壓縮匹配數。L44已證實候選前端可行；L68前端與binary lift成本尚未解決。

真正的逐層長度鏈是 L44 的 11→22→44，或 L68 的 17→34→68。
L68 的長度4壓縮是旁支相容性限制，不是4→17的壓縮展開層：
4與17沒有整除關係，不能直接套用上述巢狀壓縮恆等式。

## 重現

    .venv/bin/python experiments/research_compressed_mitm.py
    .venv/bin/python experiments/research_compressed_mitm.py --small-lengths --count-lengths --pilot-44
    .venv/bin/python -m pytest -q tests/test_compressed_mitm_research.py tests/test_compressed_pairing.py tests/test_cross_compression.py tests/test_multilevel_lift.py

第一個指令約1.4秒，只做計數與小型全枚舉；第二個約7.4秒，包含有限Z3
診斷，不進行正式長跑。時序與Z3短timeout結果可能隨機器負載變動。

測試結果：上述四個 test modules 合計68 passed（4.66秒），其中研究用
新增13項。不是宣稱整個專案的完整 pytest suite 已重新執行。
