# 正式分段佇列：中間層 MITM＋精確 GF(2) 展開

2026-09-20。將 `middle_mitm_L68_followup.md` 中已驗證的局部方法移入
正式背景 completion 佇列，不替換 GPU/C 主搜尋，也不直接枚舉全部粗壓縮。

## 啟用範圍與操作

正常執行 `main.py`、使用預設 Z3 與分段求解設定，即會啟用新方法。
首版在 **L % 8 == 4** 使用，包含 Project 2 的 **44、68**。
L46 等其他長度維持原有路徑；原 one-shot rollback 也保持不變。
只有實際取得壓縮配對工作才會執行，不是每次 GPU iteration 都呼叫。

兩個新開關預設開啟：

```text
--z3-middle-mitm
--z3-affine-lift
```

回到舊分段 Z3 方法，不需刪任何資料：

```sh
.venv/bin/python main.py --L 68 --no-z3-middle-mitm --no-z3-affine-lift
```

`--no-z3` 關閉整個 completion 分支；`--no-z3-staged` 使用原 one-shot
路徑，不使用這次的新方法。Python `SlicePolicy()` 的舊直接呼叫者仍沿用
原預設；正式 `main.py` 明確傳入上述預設開啟的選項。

## 搜尋中的位置

現有 PyTorch/GPU 與 C 搜尋照常產生候選、做精確壓縮配對；沿用已有
背景佇列、排程、矩形交換、驗證與結果去重。新的分支為：

```text
factor-4 精確配對工作
  → 既有有限次 C 矩形交換
  → 固定粗模式內的 middle MITM
      ├─ 超時／建表超限 → 原 Z3 中間層
      └─ factor-2 候選 → 去重並存成 durable child
                           → GF(2) 必要方向條件
                              ├─ 矛盾 → 該 child UNSAT
                              ├─ 空間小 → 精確完整相關檢查，分段可恢復
                              └─ 空間大 → 原 binary Z3
  → SAT：原 independent verifier → 原對稱性去重 → 原 L.txt writer
```

獨立的 factor-2 配對工作也會先使用 GF(2) 展開；半徑 repair 分支不變。
沒有改動 objective、Project 2 counting convention、GPU loss、SA 接受率、
restart 策略或既有 Z3 encoding。沒有把已找到的 PQCP 作搜尋種子。

## 精確性與可恢復性

### middle MITM

- 使用固定 coarse word 的 exact content/norm suffix DP。
- 對較小側建立完整 PACF signature 索引，保留所有碰撞項；另一側串流。
- 每個結果重新檢查 scalar PACF、壓縮映射與 parity content。
- 以既有 factor-2 canonical key（含 target/content）去除 child 等價重複。
  只略過已有 durable representative 的工作；UNKNOWN representative 留在佇列。
- JSON 保存 raw-match cursor 和 order version。每次重建有上限的索引，
  確定性重放至 cursor，預設一次取出最多 4 個不同 children 並一起保存，
  避免每個 child 都重新建表；不長期保留大型 Python 物件或 pickle。
- 重建／重放超時或 index 超限時，這個工作轉回原 Z3 enumeration。
  Z3 只 block 已保存的完整 middle pattern，不 block 未證明的區域。

### GF(2) 與精確展開

對 L=2d，c[i]=x[i]+x[i+d]，h[i]=(x[i]−x[i+d])/2：

    P_x(u)   = P_c(u)/2 + 2 N_h(u)
    P_x(u+d) = P_c(u)/2 − 2 N_h(u)

N 是 negacyclic correlation（跨尾端變號）。middle 配對後，剩餘條件為
N_ha(u)+N_hb(u)=[T(u)−T(u+d)]/4。

寫成 M 個 ±1 項的總和等於 R，負項數必須是 (M−R)/2。將項相乘就
得到方向 bits 的 GF(2) 線性必要式。線性一致不代表 PQCP，仍須檢查
完整原長度相關條件。詳細推導及窮舉佐證見前述研究報告。

- 使用 particular solution＋完整 nullspace basis；Gray 順序不重複。
- JSON 保存下一個尚未檢查的 index、rank、自由 bit 數、總空間及
  mathematical identity/order version。恢復可直接到達該 Gray index，
  不重新掃描已排除的 prefix。
- identity 包含兩側 middle、target/content 及 cross disjunction。
  不同 identity 或新 order version 會保守地從頭開始，不轉移舊證明。
- 精確整數 popcount 檢查 PACF；content 與所有指定 cross patterns 仍受約束。
- 預設 auto 使用相同 Gray 順序的 C kernel，分塊呼叫並釋放 Python GIL；
  工具鏈不可用時回退 Python。兩種實作的游標相容，沒有更改 order version。
- 超時是 UNKNOWN，游標保存後排回原佇列；空間超限才交回 Z3。
- 只有線性矛盾或全部 affine assignments 完成檢查才能 child UNSAT。
- SAT 必須通過未修改的 verifier 與 target/compression/content audit，
  再交給原 atomic persistence、等價去重及 L.txt writer。

**父問題 UNSAT 仍要求 middle enumeration 完全結束，且所有 children
都是 UNSAT。** 模型數上限、時間上限、還有一個 UNKNOWN child，都不能
宣稱父問題無解。這次不提高既有預設 16 個 middle models 上限。

## 資源與觀測

|選項|預設|超限處理|
|---|---:|---|
|`--z3-mitm-index-entries`|250,000|轉回原 Z3 中間層|
|`--z3-affine-max-assignments`|1,048,576|轉回原 binary Z3|
|`--z3-affine-backend`|auto|C 不可用時使用 Python；可指定 python 回退|
|`--z3-mitm-batch-size`|4|每次 1–16 個且不超過剩餘 model cap；1 回到逐個模式|
|`--z3-slice-seconds`|1 秒|原分段加倍／停放規則，精確展開保存游標|
|`--z3-multilevel-max-models`|16|父工作保持未解，沿用原 LIMIT 邏輯|

這些都是資源設定，不是數學剪枝。時間界線是 cooperative checks，
不是硬中止；原 Z3 建模時間仍可能增加實際 turn 時間。

新增紀錄會標明 `middle_backend`、`binary_backend`、MITM cursor／duplicates、
affine rank／assignments／examined／next_index／status。只有真正執行 Z3
check 才增加 Z3 次數，不會把 MITM 或 GF(2) 計算算成 Z3。

例如輸出：

```text
GF(2) 精確展開：child=UNSAT parent=LIFT_UNKNOWN ...
```

表示這個 binary child 已排除，但整個 coarse parent 還有待查 middle，
不是 Z3 UNKNOWN，也不是長度 L 無解。

## 程式與測試

新增：

- `solver/middle_mitm.py`：可重放、受記憶體限制的固定粗模式 MITM。
- `solver/affine_lift.py`：GF(2)、Gray 展開、完整檢查與可恢復游標。
- `tests/test_affine_lift.py`、`tests/test_exact_lift_pipeline.py`。

修改：`solver/staged_completion.py`、`solver/compressed_completion.py`、
`main.py`。實驗版程式保留原樣作為先前數據的可重現基準；正式模組不
import experiments。未刪除舊方法或改 .gitignore。

新增測試包括小型完整 ternary join oracle、逐 bit binary lift oracle、
GF(2) 模 4 條件、Gray suffix 等價性、逾時與 cap、child canonical 去重、
UNKNOWN proof ownership、舊 checkpoint 恢復、真正的 Z3 fallback、
真正的 background worker／驗證／寫檔及正確的 Z3 呼叫計數。

新增 **32 項** lightweight tests；最終其餘回歸結果為
**925 passed、8 skipped、1 warning（81.56 秒）**。warning 為既有
torch 缺 NumPy；新增的精確整數方法不依賴 NumPy。

全套 pytest 有兩個既有缺失模組阻擋 collection：
`experiments.benchmark_mps_acceleration` 和
`experiments.benchmark_torch_mathematical_loss`。未改動這些不相關缺件。
可執行其餘回歸：

```sh
.venv/bin/python -m pytest -q \
  --ignore=tests/test_mps_acceleration_benchmark.py \
  --ignore=tests/test_torch_mathematical_benchmark.py
```

## 真實背景佇列短測（隔離資料）

`results/exact_lift_integration_smoke/` 保存完整數據；原正式 L.txt 與來源
checkpoint 未更改。重放的是已保存的粗數學工作，不是從 GPU 初始化
開始的端到端搜尋，以下時間**不能當作平均找到新 PQCP 的秒數**。

- L44：原 4 次 rectangle miss 後，MITM 找到 middle，GF(2) 剩 12 個
  自由 bits；檢查 1,320/4,096 個方向後 SAT，獨立驗證通過。測試根目錄
  的 `L44/44.txt` 及完整 JSON 保存該解，實際 Z3 check 為 0。
  另外只讀比對正式 44.txt 的現有等價 key，未發現等價紀錄；結果及來源
  存 `L44/archive_audit.json`，沒有因此覆寫正式 44.txt。
- L68：兩個固定粗工作共跑 16 turns、約 4.61 秒；安全結束後再恢復
  8 turns、約 3.35 秒。原有 children 與 affine 游標保留，未重置，尚無 SAT。
  恢復後共有 7 children，其中 6 個已完整 UNSAT、1 個待解；兩個 enum
  都仍 READY，父問題沒有被誤判 UNSAT。
- 使用正式 `solve_affine_lift` 重跑先前 38 個不同 L68 子問題：全部
  完整 UNSAT，共精確檢查 12,582,912 個方向，約 25.14 秒。
  `production_affine_regression.json` 保存逐子問題輸入及結果。

這次驗證的是整合正確性、恢復能力，以及可行的局部加速路徑；尚未做
多 seed 端到端新解產率比較，不宣稱 L68 新解搜尋已提升任何百分比。

後續不改變數學條件的 C kernel／MITM batching 優化，以及最新測試與
局部效能比較，見 `docs/completion_performance.md`。
