# L86：非零位置與符號分離搜尋（2026-10-04）

已實作並接入 `main.py` 的直接三值壓縮支路。完整 PQCP 定義、binary lift、
獨立驗證與正式解庫寫入條件維持一致。目前尚無 L86 精確壓縮解或完整新解，
不宣稱提高發現新解的速度。

## 文獻查核

這次取得 Andraschko、Kotsireas、Kreuzer 的 ICMS 2026
[七頁會議稿全文](https://icms-conference.org/2026/papers/paper80/main.pdf)，
先前只有一頁摘要。它用布林變數及 Hamming weight 編碼處理相關等式，
先求三值壓縮，再 lift。表 1 比較編碼與 SAT 求解器，正文說明其測試的
大於 40 長度案例在一天內沒有完成。可借鑑編碼，不能把其效能外推至本案 L86。

「先求非零位置，再求符號」也有既有背景：Craigen (2003)
[Boolean and ternary complementary pairs](https://doi.org/10.1016/S0097-3165(03)00118-3)
的出版者摘要敘述了 support 的模 2 必要條件。此次只取得出版者摘要，
不宣稱已讀其全文。Gobin (2023)
[Boolean Complementary Pairs 碩士論文](https://mspace.lib.umanitoba.ca/bitstreams/e3f73833-0322-4661-b9ef-f91d46391501/download)
提供背景；其中零相關 Laurent 多項式因式分解不能直接套在非零週期目標。

亦查閱 Lebedev (2026-09-04)
[Quaternary Legendre Pairs of Lengths 42, 46, 52, 58, 66, 72, and 80](https://arxiv.org/abs/2609.04589)。
這是四相 Legendre pair 預印本，目標不同。其反射對稱限制沒有涵蓋所有
PQCP 的證明，故本次沒有套用此限制或其搜尋耗盡結論。

以下針對本專案目標自行推導，不宣稱是原創文獻定理。公式和實作均以
獨立整數相關重新檢查，沒有直接採用他種互補序列的零相關條件。

## 1. 壓縮目標與修復範圍

令 L=86、d=43、原始符號 x,y∈{−1,1}^86，定義

\[
a_i=(x_i+x_{i+d})/2,\qquad b_i=(y_i+y_{i+d})/2.
\]

a,b 是正規化三值壓縮符號，不是原始 0/1 bits。
P_w(u)=Σ_i w_i w_{i+u mod d}。本案壓縮目標是

\[
P_a(u)+P_b(u)=C(u),\qquad C(0)=43,
\quad C(k)=C(d-k)=\eta,\quad C(u)=0\ (u\ne0,\pm k).
\]

k=1,2，η=±1，使用既有四種 canonical content。既有小範圍修復限制
全部三值位置的 Hamming distance，符號大幅重排時可能超出該範圍。

新分支固定兩側各自符號個數，每側至多把一個非零位置移到一個零位置，
並重新決定**全部非零位置的正負號**。它不是三值半徑 4；新舊修復的
範圍不互相包含，故均保留。

## 2. 非零位置的模 2 條件

令 U_i=[a_i≠0]、V_i=[b_i≠0]。a_i≡U_i (mod 2)，因此必要條件為

\[
M_u:=P_U(u)+P_V(u)\equiv C(u)\pmod2.
\]

有一個位移不成立，便沒有任何正負號配置能在**這組固定 support** 上
達成目標。這不是充分條件，也不排除附近其他 support。

將 u=1,…,21 的全部 parity 打包為 21-bit signature。A 側建立零次／
一次移動的 signature 字典，B 側以 target XOR signature 查詢補向量。
每個 signature 保存全部不同 support，不丟掉同向量的不同位置集合。
一側非零位置數為 t 時，只需建立 1+t(43−t) 個 support，避免左右笛卡兒積。

### 精確增量公式

在 F₂ 上移動 i→j 等於 U'=U+e_i+e_j。令

\[
g_i(u)=U_{i+u}+U_{i-u}\pmod2.
\]

直接展開得

\[
P_{U'}(u)=P_U(u)+g_i(u)+g_j(u)
+[j=i+u]+[i=j+u]\pmod2.
\]

d 為奇數，只取獨立位移時，最後兩項只在
u=min(|i−j|,d−|i−j|) 對應一個 bit。因此 signature 更新為
base XOR g_i XOR g_j XOR distance_bit。此為等式，不是近似篩選。
測試對 d=3,5,7 的全部 binary support、全部合法移動，比對完整 PACF。

## 3. 固定 support：模 4 符號消元

每個非零位置放 bit s_i，符號為 (−1)^s_i。位移 u 的有效邊為同側
兩端皆非零的 (i,i+u)，共有 M_u 條。令 N_u 為有效邊中
s_i XOR s_j=1 的數量。完整相關等式**等價**於

\[
M_u-2N_u=C(u),\qquad N_u=(M_u-C(u))/2.
\]

先檢查 |C(u)|≤M_u 及 parity。取 N_u 的奇偶，得到必要線性方程

\[
\bigoplus_{(i,j)\text{有效邊}}(s_i\oplus s_j)
=((M_u-C(u))/2)\bmod2.
\]

另加入每側負號個數 n_A、n_B 的 parity：Σs_i≡n_s (mod 2)。
Gaussian elimination 給出所有解的雙射參數化 s=c+Bf。
不一致可排除這組 support 和固定 counts；一致仍需驗證完整整數邊數和
負號個數，不能只看模 4。

這是壓縮層**非零位置**的普通週期符號系統。既有 binary lift 使用
壓縮層**零位置**的 negacyclic 方向變數，包含跨界負號，兩者不可混用。

此次 archive 的 42 組及 native 的 3 組 matched support，rank 皆為 22，
43 個符號變數各剩 21 個自由變數。這是樣本實測值，不假設所有 support 如此。

## 4. 精確位元窮舉

support、負號位置各用 43-bit 整數。用 Gray code 遍歷全部自由配置，
每次只切換一個 f_j，亦即 XOR 該欄在 A、B 的兩個符號 mask。
先以 popcount 檢查每側負號個數，再逐位移驗證

```
popcount((signs XOR rotate(signs,u)) AND active_edges) == N_u
```

無亂數或浮點接受判定。free=21 時完整窮舉 2^21=2,097,152 組，
而非 2^43。native 三組 support 各自完整窮舉約 18–22 ms，另需 Python
建構和 support matching。同樣三組用目前 Z3 編碼各給 3 秒均 UNKNOWN。
此只比較這些固定 support，不是所有 SAT 編碼／求解器或 PQCP 發現率比較。

Z3 版保留供小型交叉驗證；production 使用 C。命中後以獨立 Python
整數重算全部 43 位移、counts、support，通過才建立既有 lift job。

### 狀態與完整性

- `UNSAT_FIXED_SUPPORT`：這組 support 與兩側固定 counts 沒有壓縮解。
- `UNSAT_SUPPORT_NEIGHBORHOOD`：每側至多一次非零位置移動、固定兩側
  counts 的全部 support 已排除；通過篩選者也已完整窮舉符號。
- `UNKNOWN`：assignment/support cap、取消或實驗性 Z3 timeout，不能排除。
- `SAT_COMPRESSED`：一個完整整數壓縮 witness，仍須 binary lift。

命中時目前保存首個 witness，沒有枚舉全部壓縮解。首個 witness 不能
lift 不代表其他 witness 也不能 lift。production 不根據本分支的局部
結果刪除其他搜尋工作。

## 5. 長測診斷及實驗

seed1919810、session `14e65c3af211` 的 8,760 個候選，其原始 support
syndrome 非零位移數分布為：

| 位移數 | 候選數 |
|---:|---:|
| 2 | 4 |
| 4 | 65 |
| 6 | 452 |
| 8 | 1839 |
| 10 | 2862 |
| 12 | 2409 |
| 14 | 958 |
| 16 | 163 |
| 18 | 8 |

沒有 syndrome=0，所以這批候選全部不能只重排符號，必須改動 support。

依 (energy,path,lane) 排序的前 256 個候選有 42 組 support 通過模 2，
全部 rank22。完整檢查 88,080,384 組自由配置，全部 UNSAT_FIXED_SUPPORT；
256 個指定 support 鄰域均 UNSAT。初版逐一重算 signature 約 7.09 秒，
精確增量版約 1.24 秒；status、mask、rank、assignment 數相同。
兩次 replay 並非隔離效能實驗，不能把秒數比值當作整體搜尋加速倍數。

離線 native 的 16 個候選有 3 組 support 通過，完整檢查 6,291,456 組
符號配置均 UNSAT。最低 E=4 者連模 2 匹配都沒有：每側各移動一次
非零位置再任意重排符號仍不能完成。更多位置移動、兩側 counts 改變等
範圍沒有排除。

三輪新鮮 MPS batch 使用相同 seed、交替 host 執行順序，開啟／關閉
support 分支的能量均為 8、10、10，都沒有 exact job。開啟版本共測
12 個候選、1 組 matched support，完整窮舉 2,097,152 組符號配置。
三輪新增分支耗時分別 0.0143、0.0416、0.0055 秒。這是功能及工作量
短測，不是等時間發現率實驗；並行測試與首次初始化也會影響整輪耗時。

結果：

```
results/L86_support_signs_archive/report.json
results/L86_support_signs_incremental/report.json
results/L86_support_signs_native/report.json
results/L86_support_pipeline_smoke/report.json
```

仍未找到 L86 解。新分支擴大符號重排範圍並給出明確範圍的排除證據；
是否提升新解發現率須另做獨立長測。

## 6. 預設與驗證

按 VS Code 執行 main.py，選 L86、啟用 Z3，對「直接三值壓縮搜尋」
回答 y，即包含本分支，不需新增指令。沒有啟用直接壓縮時也不執行本分支。

順序：GPU 三值 → C 整數延續 → support/sign 修復 → 既有小範圍精確修復
→ binary lift → 獨立完整驗證。新預設：

| 參數 | 預設 | 意義 |
|---|---:|---|
| `--torch-direct-compression-support-top` | 4 | 每輪候選數；0 關閉 |
| `--torch-direct-compression-support-max` | 8 | 每個候選最多求解的 support 數 |
| `--torch-direct-compression-sign-assignments` | 8388608 | 每個 support 最多自由配置數 |

每輪存 `L86_seed<seed>_support<cycle>.json`，含 reference、target、mask、
rank、free_variables、assignment 計數及結果。cycle 和 search metrics
新增 support_attempts、support_patterns、support_new_jobs、sign_assignments；
cycle 另有 support_seconds、support_statuses。子階段耗時不能重複累加。

驗證包括 d3/5/7 全部 support 增量；L6/L14 固定 counts 的完整 ternary
literal oracle 與 C、Z3 兩版逐一比對；cap、取消、不同 support 保存；
真實 pipeline 新增工作、去重、parity content 交接。全套可收集測試
1145 passed、14 skipped，另有增量公式修改後的 10 項測試全部通過。
沿用排除兩個缺少既有模組的 benchmark 測試：

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_mps_acceleration_benchmark.py --ignore=tests/test_torch_mathematical_benchmark.py
```

既有 PyTorch 缺 NumPy warning 未影響以上驗證。首個 C 編譯成本歸入
construction_seconds，停止訊號在各 support／reference 之間檢查。
