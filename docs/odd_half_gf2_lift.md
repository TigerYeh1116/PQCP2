# 奇數半長度 factor-2 展開：GF(2) 前處理＋精確 Z3

## 使用方式

直接執行 `main.py`，輸入 86 並啟用 Z3，即預設使用此方法。
適用 L=2d、d 為奇數（例如 46、86、94）；44、68 的既有路徑不變。
這只改變精確壓縮 completion，不改 PyTorch loss、FKM、SA、目標或 verifier。

啟動時會顯示「奇數半長度方向 GF(2) 前處理已啟用」。
分段求解另外記錄原始方向變數數量、rank、free_bits、前處理時間及是否真的執行 Z3。
snapshot 同樣保存上述欄位；UNKNOWN 繼續保存／排程，不視為 UNSAT。

回退舊 Z3 編碼：

```sh
python main.py --L 86 --no-z3-orientation-preprocessing
```

這個選項與 `--z3-affine-lift` 不同：舊 affine 路徑枚舉 L=4 mod 8 的小空間；
新路徑不枚舉方向空間，而是把必要線性條件消去後交給 Z3。
one-shot 和 staged completion 都支援新選項。既有 checkpoint 的數學工作與 key 不變，
恢復時按目前選項重新建立求解器；沒有改動／刪除既有 checkpoint。

## 數學依據

使用雙極符號 x，原檔案仍是 0/1。令

```
c[i] = x[i] + x[i+d]
h[i] = (x[i] - x[i+d])/2
```

非零 c 唯一決定兩個 bits；c=0 才有方向變數 h[i]=(-1)^b[i]。
由 S(d)=0，精確壓縮配對兩側合計必有 d 個零，因此 L=86 有 43 個方向變數。
這個數字以「壓縮配對已符合 folded target」為前提，不能套到任意近似候選。

令 P 為 periodic PACF，N 為 negacyclic PACF（超過長度 d 的項改負號），則

```
P_x(u)   = P_c(u)/2 + 2*N_h(u)
P_x(u+d) = P_c(u)/2 - 2*N_h(u)
R(u)     = (T(u)-T(u+d))/4
N_hA(u) + N_hB(u) = R(u)
```

**不能把 N 當成普通 periodic PACF。** 對奇數 d 有 N(d-u)=-N(u)，所以只需
u=1..(d-1)/2。已滿足 folded target 時，這些完整整數殘差方程、binary 展開
及 S(d)=0 等價於完整長度的相關目標。L=86 的獨立殘差方程有 21 個。

每個殘差方程有 M 個已知符號的方向乘積：

```
sum_e s_e * (-1)^(b_i+b_j) = R,  s_e = +/-1
```

若 |R|>M 或 M-R 為奇數，該固定工作不可能。
否則負項數是 (M-R)/2，將所有項相乘得到

```
XOR_e (b_i XOR b_j) = ((M-R)/2 mod 2) XOR (negative wrap count mod 2)
```

重複變數以 GF(2) 消去，不能丟掉原整數方程的重數。
高斯消去矛盾時只排除該固定壓縮工作；一致時，以自由變數的 XOR 表示所有方向。
若 rank=r，線性解空間有 2^(d-r) 個元素。rank 不是固定常數，不宣稱一定是 21。

Z3 仍精確檢查殘差方程的負項數，而不是只檢查 parity。
指定的 A/B 奇偶 weight 仍在還原 bits 上施加；odd d 的方向會改變奇偶 weight，不能省略。
SAT 後沿用獨立 verifier、content 與壓縮核對，再交給既有去重／原子寫入流程。

## 驗證與 benchmark

`tests/test_orientation_gf2.py` 將自由變數參數化和全方向枚舉逐一比對，
並以小型完整 binary oracle 比對新舊 Z3 可解性，檢查 content、UNKNOWN、
恢復、CLI 回退與前處理矛盾。L=86 隨機序列用實際完整 profile 核對公式；
這不等於找到 L=86 PQCP，也不等於端到端效能驗證。

```sh
python experiments/benchmark_odd_lift.py --L 14 --output results/odd_lift_small.json
python experiments/benchmark_odd_lift.py --L 46 --input-dir results/compressed_lift --output results/odd_lift_saved46.json
```

benchmark 使用交替 A/B 執行順序、相同工作與 check timeout，分別保存建模時間、
check 時間、前處理時間、rank、狀態及含完整建模的 wall time。
只在新舊都得出確定結果時比較 speedup；UNKNOWN 另外列出，不冒充 UNSAT。
保留 SAT 獨立驗證，不寫入正式 L.txt，也不讀取既有解來生成 seeds。
`--input-dir` 只讀取 snapshot 的 compressed_a/b 與 content_profile，不使用其中完整 A/B。
不提供輸入檔時只允許 L=6、10、14 的有界小型枚舉。

目前沒有 L=86 的已保存精確壓縮工作樣本；不能把小型或 L=46 子問題加速
宣稱為「L=86 找新解」的加速。之後有工作樣本可執行：

```sh
python experiments/benchmark_odd_lift.py --L 86 --input-dir results/compressed_lift --limit 12 --repeats 3 --timeout-ms 1000 --output results/odd_lift_saved86.json
```

效能測量請與 pytest／其他本機運算分開進行。沒有樣本或輸出已存在時程式會拒絕，
不重新搜尋、不覆寫舊結果。

### 本次短測（2026-09-20）

每種 L 取 12 個固定工作，各執行新舊編碼 3 次；每次 Z3 check 上限 1000 ms。

| 工作集 | 新舊確定結果 | 舊 wall 中位數 | 新 wall 中位數 | 逐對加速比中位數 |
|---|---|---:|---:|---:|
| L=14 小型 ternary catalog | 36/36 均 SAT、獨立驗證通過 | 9.540 ms | 1.930 ms | 5.03× |
| L=46 既有壓縮工作 | 36/36 均 UNSAT | 167.945 ms | 28.460 ms | 6.05× |

L=46 新編碼 rank=11，自由方向 23→12；GF(2) 前處理中位數 0.107 ms。
舊／新建模中位數為 103.588／13.181 ms，check 中位數為 63.183／15.065 ms。
逐對加速比的中位數不等於兩組時間中位數相除。
L=46 使用排序後的首 12 個不同工作，全部為 UNSAT，不代表全部工作分布。
同一個 L=14/46 小集合的重複執行不是 36 個獨立搜尋 runs。

原始結果：`results/odd_lift_L14.json`、`results/odd_lift_L46_isolated.json`。
L=46 後者是在 pytest 結束後單獨重測，較早的 `odd_lift_L46.json`
與測試收集有重疊，不用於正式效能結論。results 依專案規則不進 Git。
**這些數字不能用來聲稱 L=86 的新解發現速度提升。**
