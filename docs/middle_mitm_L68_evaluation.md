# L68 中間層 MITM＋現有 Z3：配對實驗

日期：2026-09-19。正式搜尋流程、Z3 encoding、verifier 和 L.txt writer
均未修改。實驗仍為獨立 prototype，尚未預設接入。

## 先備份

分支 `pytorch` 已建立本機快照：

    0fefc073 Snapshot search pipeline before L68 middle MITM evaluation

包含當時程式碼、測試、文件（71個檔案）；未 push。4.txt、44.txt、46.txt
與 build/pqcp_search_c 原有未提交變更留在工作目錄，沒有刪除或覆寫。
以下新實驗是在此快照之後建立，未另行 commit。

## 方法與公平性

- 只讀 `checkpoints/z3/L68_seed123` 的 factor-4 數學問題 spec，驗證 JSON
  checksum、精確 coarse PACF、長度及 alphabet。**不讀入 source_a/source_b
  作種子，不根據歷史求解時間或分數挑工作，不讀既有 PQCP。**
- 按檔名排序，使用現有完整 compressed-job key 排除等價工作，取前8個。
  問題覆蓋 k=1,2,4,17，eta 的正負及不同 parity contents。
- A：原 `solve_multilevel_lift`，包含現有 cross pruning。
- B：實驗 MITM 找 middle models，然後呼叫**同一個** `solve_compressed_lift`。
- 兩者上限皆16個 middle models；每次獨立、從零開始，不用 checkpoint
  裡先前的模型或進度。AB執行順序交替；每次只有一個工作執行。
- 子程序僅供隔離和硬性 wall-clock 停止，不是平行搜尋。計時涵蓋
  DP、枚舉、建表、cross checks、Z3建模與求解；共同的程序啟動及 import
  時間另記 startup_seconds，不計入方法預算。
- 時限中止一律 UNKNOWN，不更新正式佇列。任何 SAT 都要經完整 verifier、
  target/content/compression 重算才承認。若有解，完整 A/B/profile 存 JSONL。
- Python 3.9.6、Z3 5.1.0、macOS arm64，使用專案 .venv；沒有新增依賴。

## 中間層使用的精確優化

設 L=4n、v=c2/2∈{-1,0,1}，每個粗位置必須滿足：

    v[i]+v[i+n] = c4[i]/2

1. Suffix dynamic programming 精確計算剩餘 even sum、平方和可達值及數量。
   odd sum 由已固定普通總和與 even sum 決定，不猜未指定值。
2. 只保留兩側平方和互補的狀態：Q_A+Q_B=L/2。
3. 使用較小一側建表，另一側串流查詢。最多250,000個建表候選，超過
   只回報資源限制 UNKNOWN，不判 UNSAT。
4. ternary word 用 nonzero mask 和 negative mask 表示；每個位移精確計算：

       PACF = nonzero_overlap_count - 2*opposite_sign_overlap_count

   全部為整數 popcount，Python3.9有相容實作，不使用浮點近似。
5. 保存 u=0..n 的完整獨立 PACF signature；同 signature 的所有候選均保留。
6. middle model 必須通過 scalar PACF 及壓縮重算，再檢查所有合法 cross
   patterns，最後才交給原 binary Z3。沒有新增 Hamming 限制或放寬目標。

8個工作的 DP 後候選數（尚未逐 middle 做 cross filter）：

|工作 ID 前綴|A|B|
|---|---:|---:|
|064d8fb5ef|28,623|25,615|
|076e444ddd|28,700|13,311|
|07e9262a79|65,100|17,619|
|17dd1b2aa3|17,619|106,224|
|208880134b|41,075|7,426|
|259537c748|39,209|13,098|
|32188a737f|138,586|8,051|
|5c9215e0cb|57,020|14,477|

## 第一輪：每工作5秒，8個相同工作

|指標|原Z3中間層|MITM中間層|
|---|---:|---:|
|取得可送 binary 的 middle model|1/8|8/8|
|首個 middle model 中位時間|7個未達，不能給完整中位數|0.378秒|
|首個 middle model 最大已觀測時間|2.233秒（僅1個達到）|0.796秒|
|開始 binary completion|1|8|
|完整 verified PQCP|0|0|

8個 paired 工作的首個 middle model 都是 MITM 較早；未達不能當作
「剛好5秒達到」，也不能把兩個 reached-only median 相除當作整體加速。

## 第二輪：每工作20秒，相同8個工作

|指標|原Z3中間層|MITM中間層|
|---|---:|---:|
|取得可送 binary 的 middle model|1/8|8/8|
|首個 middle model 中位時間|7個仍未達|0.374秒|
|首個 middle model 最大已觀測時間|2.747秒（僅1個達到）|0.822秒|
|開始 binary completion|1|8|
|完整 verified PQCP|0|0|

兩種方法均把剩餘 check budget 交給當前 binary 子問題。MITM很早進入
最後一層，卻仍未產生完整解，說明這批工作最後一層也是實際瓶頸。
UNKNOWN不代表無解，也不能由此推論任何長度不存在PQCP。

## 第三輪：20秒預算＋共同1秒 binary check切片，前4個工作

並未只給MITM較有利的排程：兩種方法的每個binary check均最多1000ms。
建模另耗時間，但仍受同一外部20秒總預算約束。

|指標|原Z3中間層|MITM中間層|
|---|---:|---:|
|取得 middle model 的工作|1/4|4/4|
|開始 binary completion 的子問題|1|52|
|完成 check 並回傳 UNKNOWN|1|48|
|總時限到時尚在執行的 binary 子問題|0|4|
|binary UNSAT|0|0|
|完整 verified PQCP|0|0|

MITM每個工作13個子問題；使用現有 factor-2 canonical key 重算，52個
彼此不等價，不是同一個binary問題重複計數。**開始completion包含建模，
不把這52個說成52個已完成Z3 check。** 原Z3第一個工作在一次binary
UNKNOWN後繼續尋找下一個middle，其餘三個工作未取得middle。

短切片提升了探索的子問題數，但本次仍沒有完整解，不能宣稱提升52倍
「找到新PQCP」的速度；也沒有證據證明1秒是最佳切片。

## 結論與限制

已確認的局部改善：在這批L68工作中，中間層MITM更早提供完整、精確且
可進行binary展開的候選。5秒與20秒皆為8/8個paired first-middle wins。

尚未確認的效果：新PQCP的平均找到時間。所有方法 verified count 都是0，
不能用逾時預算代替time-to-solution，也不能估算任何可靠的加速百分比。
8個工作來自同一seed的既有佇列，並非所有L68模式的代表性抽樣。

建議保留這個可回退的局部方法，之後以相同binary子問題評估更合理的
切片／續解安排，或增加獨立來源的paired工作。不要直接替換整個GPU/SA
流程；也不要因這次短測沒有完整解，就把壓縮MITM路線判定為失敗。
目前正式預設完全不變。

## 檔案、資料與重現

- `experiments/middle_mitm.py`：獨立精確原型。
- `experiments/benchmark_middle_mitm.py`：公平配對、外部deadline、重放及統計。
- `tests/test_middle_mitm.py`：18項 lightweight tests，不跑長benchmark。
- `results/middle_mitm_L68_5s/`：inputs.json、runs.jsonl、summary.json。
- `results/middle_mitm_L68_20s/`：同上。
- `results/middle_mitm_L68_20s_sliced/`：同上；另在每次run保存全部binary
  嘗試的middle A/B、check timeout及結果，可追溯UNKNOWN。

40個限時執行共560秒求解預算，實測總方法wall time約560.07秒；程序啟動
及程式開發／單元測試時間不含在內。被外部中止的程序，其未完成check
及RSS只記最後觀測值／下界，不是精確總完成數或最終peak memory。

重放固定資料（不依賴之後改變的production queue）：

    .venv/bin/python experiments/benchmark_middle_mitm.py \
      --manifest results/middle_mitm_L68_20s/inputs.json \
      --jobs 8 --seconds-per-job 20 --output-dir results/middle_mitm_L68_replay

加切片時加 `--binary-slice-ms 1000`。Output directory必須是新路徑，避免
覆寫既有研究資料。腳本不會寫入正式L.txt或更改來源checkpoint。

## 測試

新增測試完整遍歷小型ternary字串，核對bitset PACF、DP計數、枚舉唯一性、
cross mask cache與完整join。小型真實Z3 SAT再交獨立verifier確認。
另檢查資源限制、UNKNOWN傳播、checksum、重放、censored aggregation與
兩種方法相同的binary切片上限。

此次相關5個test modules：86 passed（4.84秒）。完整 `python -m pytest -q`
在collection遇到兩個既有缺件：`experiments.benchmark_mps_acceleration`
與 `experiments.benchmark_torch_mathematical_loss`。未修改這些不相關測試，
也未安裝套件。torch缺NumPy有既有warning，與本實驗的整數計算無關。

排除上述兩個無法collection的既有test modules後，其餘回歸：

    869 passed, 8 skipped, 1 warning in 81.02s

新實驗程式與測試的 py_compile 通過。macOS 預設bytecode cache在sandbox
不可寫，因此這項檢查使用 /private/tmp 下的新暫存cache，沒有更改系統設定。
