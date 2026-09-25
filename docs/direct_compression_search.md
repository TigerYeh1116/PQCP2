# 直接三值壓縮搜尋 + GF(2) 自由變數方向搜尋

這兩個選項預設關閉，不更換 Project 2 定義、既有連續 loss、objective、
C 退火、獨立 verifier 或結果等價去重。直接按 main.py 執行鍵可互動選擇。
未讀取 L.txt、已知 PQCP 或搜尋日誌來初始化任何新候選。

## 執行

最簡單的方式是在 VS Code 開啟 `main.py`，按執行鍵，依序輸入 L、seed、
Z3 與數學 loss 選擇。適用長度會再詢問：

1. 是否啟用壓縮／方向交替 polish。
2. 是否啟用直接三值壓縮搜尋。
3. 是否啟用 GF(2) 自由變數方向搜尋。

新功能輸入 `y` 開啟，直接 Enter 或 `n` 保持關閉；開始前會顯示選擇摘要。
關閉 Z3 時只詢問第一項，後兩項會略過，不會偷偷重開 Z3。
44、68 等不適用長度會顯示說明並保留原搜尋流程，不詢問這三項。
仍持續搜尋直到 Ctrl+C，不新增預設時間上限。

以下指令方式仍保留，無須使用它才能啟動新功能。
在 Mac 的專案 .venv 中：

```sh
python main.py --L 86 --torch-device mps --z3 \
  --torch-direct-compression --z3-orientation-search
```

CUDA 主機只需把 `mps` 換成 `cuda`。保留原本持續搜尋直到 Ctrl+C 的行為。
省略兩個新選項即可回復舊路徑；也能只啟用其中一項做消融。
兩者僅支援 6<=L<=94 且 L=2 mod 4（例如 46、86、94），不會暗中用於 44/68。
CLI 必須開啟 Z3 compressed lift；直接壓縮選項限非 resume 的 PyTorch hybrid。

可調工作量（不是已證明最優的參數）：

```sh
--torch-direct-compression-batch 128
--torch-direct-compression-steps 128
--torch-direct-compression-candidates 8
--z3-orientation-search-steps 128
```

現有 PyTorch 連續搜尋與 C 退火保持不變。每次 GPU formation/polish 完成後，
增加一個有步數上限的三值 GPU 搜尋批次；這不是新的可微 loss。
方向搜尋在背景 completion worker 執行，不在 GPU producer 裡等 Z3。

## 三值層：數學與表示

L=2d 且 d 為奇數，取 v_i=(a_i+a_(i+d))/2 in {-1,0,1}。
從原目標 T 直接折疊，而不是套用 Golay 目標：

P_va(r)+P_vb(r) = (T(r)+T(r+d))/4。

r=0 強制兩條三值序列合計恰好 d 個非零、d 個零。對 L86 即 43+43。
若一條完整 binary 序列在偶/奇數位置分別有 e/o 個 1，則三值 content
可寫成 (n,z,p)，其中 z=e+o-2n，p=d-n-z，且 0<=e-n<=z、0<=o-n<=z。
因為每對 i,i+d 恰有一偶一奇，這些 content 條件也是存在此奇偶 weight
展開的充分條件（但不是存在合法 PACF 展開的充分條件）。

列出所有合法 content 配對，跨批次抽樣；不是猜定其中一組當數學定理。
每個 lane 交換同一條三值序列的兩個元素，因此普通總和、content 及上述
可展開性均保持不變。跨 cycle 改換 profile/content/初始化排列。
初始化是新產生的 fixed-content 隨機排列，不冒稱這條新支路用了 FKM；
原本主流程的 FKM/C 部分並未移除。

搜尋能量為獨立位移 1..(d-1)/2 的平方殘差和。它**不是 pqcp_objective**。
使用明確標為 heuristic 的退火溫度 4 -> 0.1，不是近似 PACF。
三值 swap 的增量式含兩位置間的交叉項；不能沿用 binary sign flip 公式。
對專案長度，float32 運算中的相關值與平方能量均在精確整數範圍內。
匯出仍以 Python 整數重算整個 compressed profile，不信任 GPU 的零分判定。

精確 match 會建構一組符合奇偶 content 的代表展開，用來交接現有 pool。
這組代表未必是 PQCP；Z3 不被限制只能找代表附近，而是求所有合法方向。
直接完整驗證已通過的代表則交給既有 writer，毋須多跑 Z3。

## 方向層：只在必要線性條件的解空間操作

沿用 orientation_gf2 的 b=b0 XOR N*z 精確參數化。在自由座標 z 中變更
一或兩個座標，可能同時改變多個實際方向 bits。所有提案先解碼，再檢查
原本 A/B 各自的奇偶 weight；不符合就拒絕，不進入局部搜尋。
每 16 個提案嘗試獨立自由指派，以免只停在某個 content-feasible 分量。
未找到第一個 content-feasible 起點前也持續獨立抽樣。全部工作有上限。

只接受不增加精確 negacyclic 殘差的合法移動。能量零後完整重算原長度
PACF，並要求 independent verifier 與指定 target 同時通過。

- 找到解：正常保存，明確標示 Z3 尚未執行，不能把觸發次數加成一次 Z3 check。
- 未找到解：若有較好的合法方向，僅作為 Z3 initial-value preference。
- 沒找到合法方向：不加任何新條件，直接進入原本 Z3。
- bounded MISS **永遠不是 UNSAT**。原有 GF(2) inconsistency 才是精確排除證據。
- 不保證固定-content 子空間在一/二自由座標移動下連通。

Staged work 完成一次方向搜尋後，將 `orientation_searched` 和合法 hint
保存到原本 atomic JSON checkpoint；同一工作再度 UNKNOWN/resume 時，
沿用 hint 而不重跑同一段方向搜尋。若在一次 turn 中途被強制終止，可能
重播該未完成 turn，與既有 staged checkpoint 語意相同。

## 保存與量測

- `results/direct_compression/L86_seed123_<job>.json`：精確壓縮工作原始資料，
  明確標記「不是完整解」，在 queue admission 之前保存。
- 同目錄 `..._cycle000000.json`：此 cycle 的提案量、最低壓縮能量、精確 job
  數、經驗證/新增解數與耗時。
- 已排入的工作沿用 `checkpoints/z3/`。原始 candidate snapshots 是稽核記錄，
  **不會自行恢復尚未排入的工作**；queue 滿仍依既有 bounded admission 處理。
- 已驗證結果仍經完整等價去重後追加 L.txt，不覆寫既有結果。
- search_metrics 增加 direct-compression 和方向搜尋計數/時間。
  方向搜尋耗時是 completion construction_time 的一部分，不可再重複相加。

## 驗證與短測

```sh
python -m pytest -q tests/test_direct_compression.py tests/test_orientation_search.py
python experiments/benchmark_direct_compression.py
python experiments/benchmark_direct_compression.py --L 46 86 --steps 2048 \
  --output results/direct_compression_extended_diagnostic.json
```

測試包含小型三值 swap 的完整枚舉、content 可展開性雙向驗證、逐步增量
profile 比對、GF(2) 方程與原 PACF 殘差一致性、literal lifts vs Z3、
MISS 回退、checkpoint 重載不重複方向搜尋，以及原 writer 的驗證/保存。

本機 CPU、兩 seed（123/456）、128 lanes、每步8 proposals 的初步結果：

| 長度 / steps | 精確壓縮工作數（兩個 seed） | 展開結果 |
| --- | --- | --- |
| 14 / 128 | 21、22 | 各取4個工作，兩種路徑一致：合計7個SAT、1個UNSAT |
| 46 / 128 | 0、0 | 未呼叫 lift |
| 86 / 128 | 0、0 | 未呼叫 lift |
| 46 / 2048 | 1、3 | 4個工作均由 Z3 判定為該工作 UNSAT |
| 86 / 2048 | 0、0 | 最低壓縮能量6、8，尚未有精確配對 |

L14 的7個SAT工作，在啟用方向搜尋時均在 Z3 前找到並完整驗證。但這不是
7個新的非等價 PQCP；benchmark 不把結果寫入正式 L.txt，也不是新解速度測試。
L46 特定工作的 UNSAT 不是 L46 無解。L86 仍無提速證據。
工具環境目前無 MPS/CUDA；保留可選方案，需在實際 GPU 上做等 wall-clock
多 seed 對照，再決定是否提高這條支路的工作比重。
