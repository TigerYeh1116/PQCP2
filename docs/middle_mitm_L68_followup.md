# L68 中間層 MITM：延長測試與精確方向縮減

日期：2026-09-20。這是獨立實驗；正式 main、solver、verifier、預設流程及
L.txt 均未因本次研究修改。沒有使用已知 PQCP 作搜尋種子。

## 結論

建議把中間層 MITM 保留為 L68 的局部加速方向，但**不要把 L44 的完整
粗層枚舉流程直接搬過來取代現行搜尋**。本次尚未找到新的 L68 PQCP，
所以不能宣稱改善平均找到新解的秒數或 restart 數。

本次有兩個可驗證的局部成果：

1. 在另一來源的四個固定粗壓縮問題上，MITM 都在 0.24–0.53 秒取得
   首個中間層候選；原 Z3 中間層只有兩個在 60 秒內取得候選。
2. 推導中間層展開方向的必要 GF(2) 方程，將這批 L68 子問題的
   34 個方向變數縮成 18 或 19 個自由變數。完整精確枚舉 38 個不同
   子問題共 25.43 秒，全部證明該固定子問題 UNSAT。

第二項是新的**實驗性精確展開診斷**，不是 Z3 測速結果，也不是找到新解
的加速倍率。這些子問題無解不代表其粗層父問題無解，更不代表 L68 無解。

## 實驗設計及觀測

沿用前輪比較程式，另取 `checkpoints/z3/L68_seed114514` 中經 checksum
驗證的數學 spec。按檔名選每個 k=1,2,4,17 的第一個工作，不根據歷史
分數、耗時或是否成功挑選，也不讀來源 A/B 作種子。固定輸入已另存。

每個工作、每種方法各 60 秒，共 480 秒；相同 binary Z3、每次 check
最多 5000ms、相同 16 個 middle model 上限，AB 順序交替且循序執行。
建表、建模及求解計入方法預算；程序啟動/import 另計。這不是 GPU/C
完整 pipeline 的端到端比較。UNKNOWN 保留為未解，不視為 UNSAT。

|k|eta|原 Z3 首個 middle|MITM 首個 middle|原 binary 啟動數|MITM binary 啟動數|
|---|---:|---:|---:|---:|---:|
|1|+1|60 秒未達|0.3415 秒|0|11|
|2|−1|60 秒未達|0.5280 秒|0|10|
|4|+1|28.8858 秒|0.3921 秒|1|11|
|17|+1|58.1621 秒|0.2345 秒|1|11|

MITM 首個 middle 的中位時間 0.3668 秒，四組 paired comparison 都較早。
原方法有兩組右設限，不能把「僅成功取得者的中位數」拿來計算整體加速。

- 原方法：2 次 binary 啟動，1 次已回 UNKNOWN，1 次截止時尚未完成。
- MITM：43 次啟動，39 次已回 UNKNOWN，4 次尚未完成。
- 兩者完整 PQCP 都是 0；所有粗層父問題仍 UNKNOWN。
- 43 個 middle 依現有完整 factor-2 key 去重後是 38 個，含 5 次等價重複。

另取四組各自的第一個未解 MITM 子問題，各給原 binary Z3 120 秒，
從零重建並包含全部相容 cross patterns。四個仍全部 UNKNOWN，共約
480 秒；不是續用前輪 learned clauses，也不是成功率的隨機抽樣。

## 為何不能直接搬用 L44 完整枚舉

令 L=4n、q=c4/2∈{−2,−1,0,1,2}。固定 content histogram h 的
必要條件包括 sum(h)=n、sum(j*h[j])=(L−2w)/2，以及兩側平方和
相加等於 fold4(T)[0]/4。L44、L68 的 n=11、17 為質數，非恆定
necklace 數可由 multinomial/n 精確計算。

在現有 canonical target/content 與必要條件篩選下，粗層每側候選數：

- L44：約 116,193–119,132。
- L68：約 620,505,404–1,220,134,042，隨 profile 改變。

這些是完整 PACF signature/PSD 篩選前的計數，不是合法解數。即使
每筆只算 24 bytes，最小 L68 側已約 14.9 GB，尚未含 hash overhead。
因此不宜直接建完整 Python 雜湊表；不代表未來分桶或其他枚舉法不可能。
特別是 k=17 的粗壓縮有 alias，平方和為 34+2*eta，不能一律用 34。

相反地，**固定粗問題內**的 middle MITM 小很多：兩個來源共 44 個
不同粗問題，較小建表側經 DP 後只有 4,751–32,726 個候選，全低於
既有 250,000 上限。這只是已檢查樣本，不是所有 L68 工作的上界。

## 展開方向數及 cross 條件

對 L=2d 的雙極序列 x，令 c[i]=x[i]+x[i+d]。
c∈{−2,0,2}；非零項固定兩個 bit，零項有兩種方向。
middle PACF 正確時，兩側 sum(c[i]^2)=T(0)+T(d)=2L=4d，
故兩側總共有恰好 d 個零項，也就是 d 個方向 bit。

- L44：2^22=4,194,304。
- L68：2^34=17,179,869,184，是前者的 4096 倍。

d 為偶數，i 與 i+d 同 parity；零項兩個方向都各含一個 1。
因此普通 weight 和奇偶 weight 在這一步不再減少方向選擇。

對 L=4n、n 為奇數，指定 mod-4 row sum 可以精確計算方向數：負的
c 固定 antipodal 兩 row 各一個 1；每個零項將一個 1 分配到同 parity
的兩 row 之一。若該 parity 有 z 個零項，其中一 row 尚需 r 個 1，
另一 row 必須需 z−r 個，方向數就是 binomial(z,r)。兩 parity、兩
sequence 相乘，再對不同的合法標記 row patterns 加總。

本批 43 個 middle 的 row-compatible 方向數最少 903,498,624、
中位 1,020,894,336、最多 2,484,332,928。它們只滿足 row 條件，
不是完整 PACF 解；僅靠已有 cross 限制仍留下很大空間。

## 新推導：未被壓縮保留的反週期通道

定義 h[i]=(x[i]−x[i+d])/2。h 在 c 非零處為 0，在 c 零處為 ±1。
令 N_h(u) 是長度 d 的 negacyclic autocorrelation：跨過尾端乘 −1。
直接展開可得（0≤u<d）：

    P_x(u)   = P_c(u)/2 + 2 N_h(u)
    P_x(u+d) = P_c(u)/2 − 2 N_h(u)

所以 middle 配對後，尚需滿足的精確條件是：

    N_ha(u) + N_hb(u) = R(u) = [T(u)−T(u+d)]/4

不是對 h 再套普通 cyclic PACF。N(d−u)=−N(u)，d 偶數時
N(d/2)=0，因此 L68 有 16 個非平凡位移代表，L44 有 10 個。
這裡只計 symmetry 後的方程，不宣稱所有二次方程代數獨立。

### 模 4 導出的線性 GF(2) 必要條件

固定一個位移，殘差方程可以寫成：

    sum_e s_e * y_i * y_j = R，總共 M 項
    s_e ∈ {−1,+1} 為已知 wrap 符號，y_i = (−1)^b_i

必要條件為 |R|≤M、M−R 為偶數，而且負項恰好 (M−R)/2 個。
把全部項相乘，可得精確的線性必要方程：

    XOR(出現奇數次的 b_i)
      = ((M−R)/2 mod 2) XOR (負 wrap 符號數 mod 2)

GF(2) 消去若矛盾，可安全排除此固定 middle。若相容，只是必要條件，
仍需完整 PACF。實驗建立 particular solution 與完整 nullspace basis，
用 Gray 順序逐一遍歷所有剩餘方向，再用整數 bitset/popcount 檢查
完整相關條件。SAT 必須通過原獨立 verifier；只有全部遍歷完才能 UNSAT，
時間不足回 UNKNOWN。没有近似，也沒有因資源上限宣稱無解。

## 精確枚舉結果

四個原本 Z3 各 120 秒未解的子問題，首次實驗約 0.49–0.98 秒各自
完整排除。之後對 38 個不等價子問題全測，補強測試後又完整重跑一次：

|指標|最後一次重跑|
|---|---:|
|GF(2) rank=16，剩 18 個自由 bit|28 個|
|GF(2) rank=15，剩 19 個自由 bit|10 個|
|完整遍歷方向組合|12,582,912|
|固定子問題 UNSAT / SAT / UNKNOWN|38 / 0 / 0|
|每子問題最小 / 中位 / 最大秒數|0.508 / 0.538 / 1.137|
|合計秒數|25.430|
|新完整 L68 PQCP|0|

每個 UNSAT 記錄均核對 enumerated == affine_assignments。這些證明
只涵蓋固定 middle 的全部 binary lifts；粗問題仍有其他 middle 未查，
不能關閉父問題，也不能估算新解平均找到時間。

另有小型反例驗證「壓縮及 row 可行不代表能展開」：L12、k3、eta−1，
middle A=(−2,−2,0,−2,2,0)、B=(−2,0,0,2,0,0)。有 32 個符合
row 條件的方向，但全部 64 個原始方向都沒有完整解。B 的 N(1)=0
要求 h1*h2*h4*h5=−1，N(2)=0 卻要求同一乘積為 +1，直接矛盾。
L68 的上述 38 個則是線性系統相容，最後由完整相關檢查全部排除。

## 建議接入方式（本次尚未修改正式流程）

保留現有 GPU/C 產生粗候選，再對固定粗問題使用 bounded middle MITM。
後續先做精確方向 GF(2) 縮減：小 affine 空間可直接精確展開，較大或
未完成者保留給 Z3。這比直接把更多 middle 丟進同一慢速 binary Z3
更值得進行下一輪受控實验，但是否提升新解產率仍待驗證。

接入前應具備：middle canonical 去重、持久 child queue/枚舉游標、
UNKNOWN 保存、資源超限的 fallback。必須遍歷完全部 middle 且每個
child 都 UNSAT 才能宣稱父問題 UNSAT。資源限制是排程，不是數學剪枝。

## 檔案、重現及測試

新增 `experiments/analyze_middle_mitm_scaling.py`、
`tests/test_middle_mitm_scaling.py` 及本文。既有正式 solver 未修改。

資料保存在：

- `results/middle_mitm_L68_followup_design/`：固定輸入、計數及摘要。
- `results/middle_mitm_L68_followup_60s/`：四組 paired 比較。
- `results/middle_mitm_L68_binary_120s/`：四組延長 Z3 診斷。
- `results/middle_mitm_L68_affine_probe/`：四組方向縮減試驗。
- `results/middle_mitm_L68_affine_all/`：38 組首次完整枚舉。
- `results/middle_mitm_L68_affine_recheck/`：38 組最終完整重跑。

重現時 output directory 必須用新路徑，避免覆蓋既有證據：

```sh
.venv/bin/python experiments/benchmark_middle_mitm.py \
  --manifest results/middle_mitm_L68_followup_design/inputs.json \
  --jobs 4 --seconds-per-job 60 --binary-slice-ms 5000 \
  --output-dir results/L68_followup_replay

.venv/bin/python experiments/analyze_middle_mitm_scaling.py \
  --probe-manifest results/middle_mitm_L68_followup_design/deep_inputs.json \
  --seconds-per-child 120 --output-dir results/L68_binary_replay

.venv/bin/python experiments/analyze_middle_mitm_scaling.py \
  --probe-manifest results/middle_mitm_L68_followup_design/all_middle_inputs.json \
  --seconds-per-child 10 --enumerate-affine \
  --output-dir results/L68_affine_replay
```

新增 15 項測試通過：包含所有小型 binary 的通道恆等式、row 計數、
GF(2) 與暴力法比較、Gray 展開逐 bit 獨立交叉驗證、40 個小型
middle 子問題的完整 binary 暴力 oracle，以及 SAT/verifier/timeout。
七個相關模組合計 **143 passed（7.18 秒）**。

完整 `python -m pytest -q` 仍受兩個既有缺件阻擋 collection：
`experiments.benchmark_mps_acceleration` 與
`experiments.benchmark_torch_mathematical_loss`。沒有刪改不相關測試。
明確忽略這兩個對應 test modules 後：
**893 passed、8 skipped、1 warning（82.57 秒）**。
warning 是既有 torch 缺 NumPy，這些精確整數實驗不依賴 NumPy。
