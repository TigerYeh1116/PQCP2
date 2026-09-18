# 保持雙向壓縮的矩形交換搜尋

## 接到哪裡

正式 `main.py` 的 staged Z3 completion 預設加入一條**獨立 C 支線**：

```
factor-4 精確匹配 / canonical 工作去重
→ 列出所有相容的 factor-n 模式
→ 構造固定行列和的 4×n binary matrices
→ 有限次 C 矩形交換搜尋（每個 restart 換模式）
→ 若未解：原本的多層 Z3 → 完整 binary lift
→ independent verifier → 原 L.txt 去重 writer
```

適用 L=4n、n≥3 且為奇數：L=44 為 4×11、L=68 為 4×17。
L=46 等不適用的長度不變。主 PyTorch／C 退火依然繼續產生其他
factor-4 工作，沒有把整個搜尋鎖死在一組壓縮模式內。
沒有使用既有 PQCP 作為 initializer，沒有修改主搜尋的 loss／neighborhood。

這是背景 completion 的局部搜尋，不是把主 C 退火換成矩形交換。
使用 `ctypes.CDLL` 執行 C 核心會釋放 GIL；不增加 GPU 相依套件。

## 精確索引與不變量

矩陣格 `(r,c)` 對應唯一的完整序列位置：

```
i mod 4 = r
i mod n = c
i = c + n * ((r-c) * n⁻¹ mod 4)
```

不能使用 row-major 排列，也不能以相鄰 positions 展開。
binary bits 與符號仍使用 0→+1、1→−1。

column c 的 bit sum 為 `(4−c4[c])/2`；row r 的 bit sum 為
`(n−cn[r])/2`。切換

```
1 0       0 1
0 1  ↔    1 0
```

使每個受影響 row/column 都維持一個 1，故每條序列的
factor-4、factor-n 壓縮、總 weight、偶數 weight、奇數 weight 全部不變。
奇偶 weight 是 rows 0+2 與 rows 1+3 的和；不要求選到的兩個 row 同 parity。
**一般不保持 factor-2 壓縮，也不保持完整 PACF**，不可混淆。

initializer 使用通過既有 Gale–Ryser 判定的 labeled margins，依 column
degree 遞減、每次配置給最大 remaining-degree rows，平手由 deterministic RNG
決定。它產生可行矩陣，但不宣稱均勻抽樣所有固定行列和矩陣。

## 目標能量與增量更新

使用既有 `structured_energy.full_error_energy` 定義：

```
T(0)=2L, T(k)=T(L-k)=4η, 其餘為 0
E_full = Σ[u=1..L/2] (S(u)−T(u))²
```

整數原始能量，**不是終端以前的 pqcp_objective score，也不是除以 16 的值**。
兩種已滿足的壓縮殘差恆為零，本支線不需要再次加其 heuristic 權重。

設四個 flipped positions 為 F，符號序列為 x，則

```
ΔP(u) = −2 Σ[i: (i∈F) XOR (i+u∈F)] x[i]x[i+u]
```

只計一端 flipped 的 directed cyclic terms；兩端同時 flipped 的項不變。
從每個 flipped position 檢查 outgoing/incoming edge，即可 O(L) 精確更新；
u=0 不變，u=L/2 的兩個 directed terms 按原 PACF 定義保留，不誤除以二。

每步抽樣多個合法 rectangles 選較低能量者，再依既有 C 搜尋的溫度家族
`6*(1−move/max_moves)+0.15`、Metropolis scale=64 決定是否接受。
這些是 heuristic，不是數學 pruning。沒有添加 tabu、矩形交換以外的 neighborhood
或改動原本退火 acceptance。

## 預算、restart、保存與 Z3

- 預設每個 factor-4 job 最多 4 個 rectangle restarts。
- 每 restart 最多 1000 moves，每步評估 8 個合法矩形，可能重抽同一矩形。
- 用 `(seed, raw job identifier, restart)` 推導 RNG，不消耗主搜尋的 RNG。
- 將相容 cross-pattern 列表 deterministic shuffle 後循環選取；未跑完一輪
  不重複選 pattern。若只有一組，會在相同 margins 下更換 initialization。
- 每輪結束即保存其進度，以及整個 job 的 best A/B、完整 profile、energy。
  Ctrl+C 等當前有限 C 輪結束；若程序意外中斷，未提交的一輪可確定性重做。
- 原中間層被 model cap 暫停的舊工作，也可恢復尚未執行的矩形搜尋；
  這不會清除已證明的 Z3 branches 或提高原本 model cap。

E_full=0 必須通過 Python 完整重算、兩種壓縮、content、指定 target 與
原 independent verifier，才可寫入；不為了「認證」而浪費一次 Z3 呼叫。
`RECTANGLE_SAT` 表示已完整驗證，`RECTANGLE_MISS` 只表示有限搜尋未找到。
沒有合法矩形也不是全域無解；本支線從不產生 UNSAT。

miss 後，原本完整的 Z3 枚舉照常。C best bits 可透過 Z3
`set_initial_value` 提供 orientation variables 的初始偏好；它**不是**
`solver.add(bit == value)`、不是 Hamming 半徑，也不固定 C 曾選過的 cross pattern。
若 Z3 版本沒有此提示 API，保持原本求解而不加提示。即使 C best 的 factor-2
模式與 Z3 本輪模式不同，也只是可違反的偏好，不會刪除任何 exact solution。

矩形 moves/evaluations 是另列的支線統計，不混進主 C 的雙 bit swap counters。
只有真的執行 Z3 check 才增加 Z3 使用次數。

## 檔案與回退

- `csrc/rectangle_search.c`：合法矩形抽樣、精確 deltas、bounded local search。
- `solver/rectangle_search.py`：CRT、margins initializer、C build/load、獨立 audit。
- `solver/staged_completion.py`：模式 restart、持久化與 Z3 交接。
- `solver/compressed_lift.py`：只增加可選的軟偏好，不改 exact constraints。
- `tests/test_rectangle_search.py`：窮舉 margins、delta、連續更新與整合測試。

首次使用自動以系統 `cc` 編譯小型 shared library 到 `build/`，不重寫
原本 `pqcp_search.c`。macOS 使用 `.dylib`，Linux/Colab 使用 `.so`。
不須安裝 NumPy／額外 Python 套件。

```
--no-z3-rectangle-search       # 只關閉矩形支線與其 Z3 偏好
--z3-rectangle-restarts 4
--z3-rectangle-moves 1000
--z3-rectangle-samples 8
```

此功能目前接在預設 staged + multilevel + cross-compression 路徑；
關閉其中路徑或停用 Z3 時，不會執行此支線。主 PyTorch/C 搜尋照舊。
沒有變更現有正式 L.txt，測試使用暫存目錄。

## 驗證及效果限制

窮舉 4×3 labeled margins 比對完整 binary matrices；連續 random rectangle
每步與完整 PACF 重算比對（含 L=44、68）；C 多步更新、雙向壓縮、
content、deterministic resume、Z3 hint 非硬限制均有測試。

短測從非 PQCP 來源完成並獨立驗證 L=12。
L=44 的一個真實匹配 job：每輪 1000 moves／8000 evaluations，
第一輪 E_full 1456→96，四輪後仍 miss，確實交回 Z3。
另以現場代數構造的壓縮測試模式（非既有 PQCP）驗證 L=68：
第一輪 4000→480，四輪最佳 384，仍不是 PQCP。
這個代數測試 fixture 未加入正式候選生成方法。

以上證明實作與交接正確、能降低部分候選殘差，**不證明 L=44／68
首解速度提高**。固定壓縮可能本來就沒有 lift；新模式來源與原 Z3
仍須保留。更長時間的公平比較應另設對照，不因短測 miss 就判定無效。
