# 長度 86 PQCP：數學分析與局部精確修復實驗

日期：2026-10-01。結論：現有紀錄的主要瓶頸是產生**完全符合壓縮相關目標**的候選。新增的半徑 4 精確修復能快速判定一個有限鄰域，但尚未找到 L86 PQCP，也尚未證明端到端搜尋成功率提高。後續已接入 `main.py` 的直接壓縮支路，啟用該支路時預設修復最多 8 個候選；詳見 [整合說明](direct_compression_search.md#2026-10-01-半徑-4-精確修復整合)。

後續長測 `results/search_metrics/L86_seed789_1a2548f80265.json`：33259.013 秒（約 9.24 小時），79167488 proposals、0 exact jobs、0 unique matches；worker waiting 33258.977 秒。604 個 cycle 最低能量分布為 {8:8,10:38,12:110,14:212,16:175,18:57,20:4}。此長測仍使用整合前版本，不能拿來當修復器的成效評估。

## 1. 原題與必要条件

`文獻/06_PACP.pdf` 第 29 頁 Project 2 第三題要求兩個非零旁瓣，各自絕對值為 4。以下所有索引取模，令二元符號序列 x,y∈{−1,+1}^86，P_x(u)=Σ_i x_i x_(i+u)，S=P_x+P_y。目標為

```
S(0)=172;
S(k)=S(86-k)=4η, η∈{−1,+1}, 1≤k≤42;
其餘 S(u)=0，包括 S(43)=0。
```

相關函數具有反射對稱，所以兩個非零旁瓣必成上述一對，符號相同。這比「最大旁瓣不超過 4」嚴格。

設 X=Σx_i、Y=Σy_i、U=Σ(−1)^i x_i、V=Σ(−1)^i y_i。直接交換求和得

```
X²+Y² = 172+8η;
U²+V² = 172+8η(−1)^k。
```

因此 η=+1 時 (|X|/2,|Y|/2) 是 (6,3) 或其交換；η=−1 時是 (5,4) 或其交換。交替和也受對應限制。可逆抽取 i↦qi、gcd(q,86)=1 將奇數 k 化至 1，偶數 k 化至 2。配合既有符號、交換等正規化，程式目前有以下四種 canonical target/content profiles；這些是**既有剪枝**，不是本次新發現：

| k | η | A 偶位 bit=1 | A 奇位 bit=1 | B 偶位 bit=1 | B 奇位 bit=1 |
|---|---|---|---|---|---|
| 1 | −1 | 16 | 22 | 18 | 21 |
| 1 | +1 | 16 | 21 | 18 | 22 |
| 2 | −1 | 17 | 21 | 17 | 22 |
| 2 | +1 | 17 | 20 | 17 | 23 |

bit=1 代表符號 −1。來源：`solver/target_profiles.py`。

## 2. 實際瓶頸

讀取既有 `results/search_metrics/L86_seed123_3e320721f1d0.json` 與 `L86_seed456_8d0777376cf0.json`：

| seed | 記錄運行秒數 | direct compression proposals | exact compressed jobs | matching unique matches |
|---|---:|---:|---:|---:|
| 123 | 10039.962 | 22675456 | 0 | 0 |
| 456 | 6420.455 | 15073280 | 0 | 0 |

合計約 4.57 個程序小時、37748736 次 direct proposals。若程序重疊運行，不能將此當作實際經過的牆鐘時間。兩份紀錄中 completion worker 幾乎全程等待，沒有求解工作。這支持優先改善 coarse candidate 的產生，而不是先優化 lift 的矩陣運算。

現有 288 份 direct-compression cycle 摘要各自只跑 128 steps；最低能量為 8，未保存非零能量候選的完整序列。新實驗保存完整 reference，才能重現並精確判定鄰域。延長單次搜尋、保留候選池屬於待測的排程啟發式，不能由這些紀錄直接推論一定較快。

## 3. 43 維三值方程

令 a_i=(x_i+x_(i+43))/2、b_i=(y_i+y_(i+43))/2，則 a,b∈{−1,0,1}^43。將原相關和依 mod 43 分組可得

```
P_a(r)+P_b(r) = [S(r)+S(r+43)]/4 = C(r)。
C(0)=43; C(±k mod 43)=η; 其餘為 0。
```

於是兩個三值序列合計恰好 43 個非零位置、43 個零位置。若一側原二元序列偶、奇位的 bit=1 個數為 e,o，三值 −1 個數為 n、零個數為 z，則

```
z=e+o−2n;  +1 個數=43−n−z;
0≤e−n≤z, 0≤o−n≤z。
```

原因是相距 43 的兩個位置奇偶相反；每個 −1 消耗偶、奇位各一個 bit=1，每個零恰消耗其中之一。這些條件保證內容可按指定奇偶權重展開，**不保證展開後相關正確**。仍須現有 exact binary lift 與獨立整數驗證。

86=2×43，43 是質數。標準因數壓縮走到 43 後沒有非平凡的下一層；壓成長度 2 主要重現奇偶和限制。因此不可直接期待多層壓縮在其他長度的收益。普通 periodic Golay 的零旁瓣目標更不適用：它要求 (X/2)²+(Y/2)²=43，沒有整數解。

## 4. 已實作：固定內容的精確修復

### 偽布林方程

`solver/compressed_repair.py` 為每個三值位置設互斥布林變數 p_i,n_i，令 v_i=p_i−n_i，故

```
v_i v_j = [p_i∧p_j]+[n_i∧n_j]−[p_i∧n_j]−[n_i∧p_j]。
```

用加權布林等式強制 21 個獨立相關方程，以符號個數約束強制零位移方程，再限制與 reference 的 Hamming 距離。所有係數、約束與回傳驗證均為整數；沒有以浮點近似排除解。

### 半徑 4 等價於至多兩次交換

固定每側三種符號的個數。對每個改動位置畫一條「舊符號→新符號」有向邊。內容不變意味每個符號頂點的入度等於出度，因此邊可分解為有向環。只有三個頂點且沒有自環，簡單環只可能長 2 或 3。

當兩側合計至多四個位置改變時，只可能是：零改動、一個 2-cycle、一個 3-cycle、或兩個 2-cycles。一個 2-cycle 是一次交換；一個 3-cycle 可用兩次交換實現；兩個 2-cycles 也是兩次交換。反之兩次交換最多改變四個位置。因此：

**每側固定內容的 Hamming 半徑 4 = 每側內交換、兩側合計至多兩次的所有候選。**

這個論證依賴三值字母表，不能直接推廣到四值字母表。也不涵蓋改變各符號個數的候選。

`csrc/ternary_repair.c` 完整枚舉上述交換；`solver/ternary_repair_native.py` 負責輸入檢查、編譯載入及獨立結果驗證。交換 p,q，設 t=v_q−v_p，對 u>0 的精確相關增量為

```
ΔP(u)=t[v_(p+u)+v_(p−u)−v_(q+u)−v_(q−u)]
       −t²([p+u=q]+[q+u=p])。
```

此式由 P(v+δ) 展開得到，δ_p=t、δ_q=−t。第二次交換必須在第一次交換後的序列計算增量。候選任一方程不等即可提前拒絕，並不漏解。

狀態含義：`SAT_COMPRESSED` 只證明壓縮目標成立；`UNSAT_LOCAL` 只排除該 reference、該內容與指定半徑；`UNKNOWN` 包括逾時或枚舉預算耗盡，不提供不存在證明。編譯錯誤或驗證失敗會丟出例外，不會當成 UNSAT。

## 5. 實驗與測試

環境為本機 M6/32 GiB、PyTorch 2.8.0 MPS、Z3 5.1.0.0；CPU C kernel 使用 cc -O3。`experiments/benchmark_compressed_repair.py` 從隨機三值候選開始，不讀現有 PQCP 當種子，也不寫正式 L.txt。

第一輪 seed=123,456，各 128 lanes、2048 steps，每個 seed 選能量最低兩個候選。GPU 生成各約 1.94、1.81 秒，最低能量分別 6、8，沒有 exact compressed job。四個 reference 的能量為 6,8,8,8。

| 同一批 reference 的局部工作 | 結果 |
|---|---|
| Z3，半徑 4 與 8，各 5 秒，共 8 次 | 全部 UNKNOWN |
| C 完整半徑 4，各一個鄰域 | 四個全部 UNSAT_LOCAL |
| C 各鄰域交換候選計數（包含重複終點） | 1242110、1169642、1169642、1019090 |
| C 暖機後，含 Python 包裝的重複測量 | 約 3.99–4.37 ms / 鄰域 |

重複測量第一次約 36.87 ms，含 target validation cache 冷啟動；當時動態庫已編譯，不能當作首次編譯總時間。Z3 未完成而 C 完成，不能把 5 秒/4 毫秒宣稱為 PQCP 搜尋成功率的倍數。四個 reference 太少，尚不足判斷此修復器對找到完整解的效益。

原始資料：`results/L86_compressed_repair_probe/report.json`（初版全部使用 Z3）、`native_radius4.json`、`native_radius4_repeated.json`。程式目前半徑 4 預設改用 native；重現初版請指定 `--radius4-backend z3`。每次輸出路徑須不存在。

```sh
.venv/bin/python experiments/benchmark_compressed_repair.py --output results/L86_repair_new --radii 4 8
.venv/bin/python experiments/benchmark_compressed_repair.py --output results/L86_repair_z3_new --radius4-backend z3
.venv/bin/python -m pytest -q tests/test_compressed_repair.py
```

新增 8 項測試：L6/L14 固定內容完整枚舉 oracle 比對、SAT 結果獨立檢查、預算耗盡與 UNKNOWN 語義、非法輸入，以及小型字母表完整枚舉驗證兩交換等價定理。全套可收集測試：1106 passed、12 skipped、1 warning，約 50.69 秒。沿用先前環境的兩個排除項，因其引用的 benchmark 模組不存在：

```sh
.venv/bin/python -m pytest -q --ignore=tests/test_mps_acceleration_benchmark.py --ignore=tests/test_torch_mathematical_benchmark.py
```

warning 是 PyTorch 提示未安裝 NumPy，未影響上述測試。

## 6. 其他有數學依據的方向

1. **交替調變縮減存在性分支。** 同時令 x'_i=(−1)^i x_i、y'_i=(−1)^i y_i，得到 S'(u)=(−1)^u S(u)。因此奇 k 的 η=±1 互相對應，偶 k 不變。只找存在性時四種目標可先搜三種，再恢復另一符號。現有解庫等價定義未納入此變換，不能直接當作既有去重規則，也不能省略結果恢復。本次未變更正式正規化。

2. **改成長度 43 的四序列與交叉相關匹配。** 定義 x_(2i)=a_i、x_(2i+1)=b_i、y_(2i)=c_i、y_(2i+1)=d_i（此節 a,b,c,d 為二元符號，不是前節三值壓縮）。若 R_ab(r)=Σ_i a_i b_(i+r)，則

   ```
   S(2r)=P_a(r)+P_b(r)+P_c(r)+P_d(r);
   S(2r+1)=R_ab(r)+R_ba(r+1)+R_cd(r)+R_dc(r+1)。
   ```

   奇 k 的目標要求四序列在非零 r 上互補。可先生成這種 quadruple，再用保持各自 PACF 的旋轉、反轉、符號變換，對奇位移交叉相關做精確簽章匹配。上述是恆等式；是否能高效生成足夠 quadruples 尚待實作和測試。不能未經證明強制四序列都對稱，否則會把一般問題縮成特殊子類。

3. **避免錯誤套用乘法軌道限制。** 若要求兩側都對某個乘法單位 q 不變，則相關目標也須不變。因此 q{±k}={±k}。對 k=1 或 2，均強制 q=±1 mod 86。故不能用非平凡的 3、7、21 階乘法子群同時固定兩側，套用其他差集問題的軌道構造；這種限制與此稀疏目標不相容。此論證針對兩側各自固定，不排除允許交換兩側等不同群作用。

下一階段優先測量「持續的壓縮搜尋＋保存不同低能量 reference＋半徑 4 精確檢查＋較大半徑 PB/SAT」。精確檢查只排除已證明無解的局部集合；重啟、候選挑選、時間分配仍是啟發式，不承諾完整性或發現時間。應以 exact compressed jobs/秒、可 lift 比率及獨立驗證的新解數作指標，而不只看 GPU 使用率。

## 7. 近期文獻與可移用範圍

| 來源 | 本次閱讀範圍與相關內容 | 對本題的限制 |
|---|---|---|
| Andraschko、Kotsireas、Kreuzer，2026-03-04，[Computing Periodic Golay Pairs Using SAT Solvers](https://icms-conference.org/2026/abstracts/Andraschko.pdf) | ICMS 一頁摘要；先解 {−2,0,2} 壓縮序列，再 lift；提出 Boolean circuits / Hamming weight / adder encodings | 原文目標為 periodic Golay，須改寫為本題稀疏 target；本次 PB 編碼與兩交換證明另行推導，未聲稱重現其完整實驗 |
| Lumsden、Kotsireas、Bright，[New Results on Periodic Golay Pairs](https://arxiv.org/abs/2408.15611)，2024 首稿、2025-03-16 v3 | 阅读 v2 HTML 與版本頁；compression、multilevel compression、orderly generation | 一般搜尋原理有用；86 的質數半長與 PQCP 目標不同，不能搬用 PG 特有結論 |
| Perera、Kotsireas，2025，[A low-complexity algorithm to search for Legendre pairs](https://www.sciencedirect.com/science/article/abs/pii/S0024379525000102) | 出版頁摘要；DFT 結構與 FFT/Bluestein 加速思路 | 未取得全文，不能聲稱其測試收益會轉移到長度 43、MPS 或完整 PQCP 流程 |
| Huang、Hsu、Liu、Chen，2026，[A Novel Design of Sparse Vector Coding Aided URLLC With Periodic Quasi-Complementary Pairs](https://repository.essex.ac.uk/42886/)；[接受稿](https://repository.essex.ac.uk/42886/1/Final%20Version.pdf) | 查核 Definition 7 與構造；其 PQCP 定義為旁瓣幅度上界 | 上界限制不等於本題恰兩個非零旁瓣；不能因名稱相同推論 L86 已解 |

查閱日期 2026-10-01。本次檢索與有限實驗都不構成「文獻中不存在 L86 解」的證明。
