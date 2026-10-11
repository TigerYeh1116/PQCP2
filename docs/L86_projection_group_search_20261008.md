# 共用 C 壓縮目標的搜尋：53 個完整內容案例合併為 27 組

2026-10-08。這是新增、已接上獨立支撐抽樣器的精確流程。
它加快相同完整目標組的判定，目前沒有找到新的完整 L86 PQCP，
所以仍不能宣稱已達成「新解發現速度提升至少 20%」。

## 1. 完整目標與壓縮目標不同

令完整目標只在 $k,86-k$ 有值 $4\eta$，其餘非零移位皆為 0。
在 odd-half indexing 下，C 與調制後 G 的目標分別為

\[
N_{C_A}+N_{C_B}=43\delta_0+\eta(\delta_k+\delta_{43-k}),
\]
\[
N_{G_A}+N_{G_B}=43\delta_0+(-1)^k\eta(\delta_k+\delta_{43-k}).
\]

把完整位置 $k$ 換成 $43-k$，C 的整個 profile 不變，G 的符號反轉。
兩個位置一奇一偶，對完整長度 86 分別有 gcd 1、2，
所以不能用完整 unit decimation 把它們當成同一個目標軌道。
尤其不能因原 G-UNSAT，就排除 sibling 的 G。

原先完整代表位置取 1、2。偶數 unit orbit 也可用 42 代表：
$21\cdot2\equiv42\pmod{86}$，而 21 是 odd unit。
相應序列 decimation 是索引乘其逆元 41。
Odd unit preserves even/odd contents，所以必要內容資料不變。
用完整代表 1、42 後，兩者的 C 目標都在 half-length 的 $\pm1$。

這是完整目標 orbit 的合法換代表，不是固定中心球的座標裁剪。
支撐抽樣只固定 contents，全部 literal 支撐仍可產生。

## 2. 內容合併

目前證明支撐下界與 binary-content bounds 留下 53 個 canonical cases。
以「全部 folded C profile、兩列 negative/zero/positive counts」作精確 key，
得到 27 組：26 組含兩個完整目標，1 組只含偶數目標。
所有原始 53 個 cases 都保留，其轉移 unit 另存 `projection_groups.json`。

單獨的一組不是任意漏掉奇數目標。
它的 negative count 為 $n_{C_A}=15$，奇數目標已被既有完整 binary-content 證明排除，
偶數目標仍保留；合併沿用原先已證明的內容清單。

給定 C contents 及 literal supports，C 的 exact signed norm 問題與完整 target parity 無關。
G supports 為補集，各 target 的 G norm 與 negative counts 都重新建構。
原始 sign quota、全部整數 norm 及重建後 binary parity contents 都必須滿足。

## 3. 精確布林分支

令 $C$ 表示固定支撐的 C sign problem 有解，$G_i$ 表示第 $i$ 個完整目標的 G sign problem 有解。
同一組至少存在一個完整解，正好等價於

\[
C\land(G_1\lor\cdots\lor G_t).
\]

所以整組完整 UNSAT 只能來自

\[
\neg C\quad\text{或}\quad\bigwedge_i\neg G_i.
\]

只證明某個 G-UNSAT 不足以排除整組。
CBC UNKNOWN、G UNKNOWN 或尚未拜訪的 branch 都不能拿來作 UNSAT。
若一個 G 已 SAT、C 尚未決定，仍是 UNKNOWN。
若 C 已完整 UNSAT，即使 G 未求解，也已正確排除全部目標。
若所有 G 都已完整 UNSAT，也不用重做 C。

實作先建立共享 C plan 與各 G plan，以 exact affine assignment counts 排序。
排序僅影響處理先後，任何順序下都使用上述同一邏輯規則。
C native 只求一次；其中間見證經獨立整數 norm/count/support 檢查。
各 G 見證同樣獨立核對，且只與自己實際的 full target 重建。
完整 binary verifier 與指定 parity content 再核對後，才回傳或保存 SAT。

程式：`solver/projection_group_targets.py`、`solver/projection_group_completion.py`。

## 4. 原流程回退與完整配對

`--projection-groups --shared-c-completion` 使用共用 C 的新邏輯。
`--projection-groups --no-shared-c-completion` 對相同 literal 支撐與相同目標組，
逐一呼叫原本 `complete_pqcp_supports`，當作參考回退。
兩邊都在首個真正完整 SAT 時停止該支撐，沒有讓回退做額外的 SAT 後工作。

`results/L86_projection_group_pipeline_paired_20261008/report.json` 已完成。
三次各 65,536 組，種子 123、456、789，交替執行順序，來源與初始 inventories 凍結。
所有支撐、目標組、contents 與完整決定逐行一致。
CBC/G 中間見證重新核對，UNKNOWN=0、完整新解=0。
中位數由 **42.240123 秒** 降至 **29.098967 秒**，同一組搜尋域的耗時減少 **31.1106%**。
這是同一支撐與同一組完整目標的實測，不是用不同區域或較少目標比較。
但它仍不是已量得的「完整新等價類發現速度」；兩邊均沒有實際新類別。

小長度 6 使用全部原始 binary pairs，比對每一組 contents、literal supports、完整 SAT/UNSAT，
以及 budget=1 的保守 UNKNOWN。
長度 86 的測試核對 53→27 合併、每個 unit、全部 parity contents 與相同 folded profiles。

## 5. 其他長測與來源完整性

一千萬次原獨立抽樣已完成：seed 314159265，3167.717594458 秒，
9,910,972 個完整固定支撐 UNSAT、89,028 個固定第一列 weight family 為空、UNKNOWN=0，
398 個 C 見證、完整新類別=0。
398 個見證的 sibling full targets 也另外全部求解：8.482028 秒，全部決定，仍無新解。
這不是將原 G 失敗複製成 sibling 失敗。

41 個先前 C 見證的一階 L1 鄰域已完整結束：5,371 個 signed C_A、451,164 個 targets、
UNKNOWN=0、完整新解=0。source C_B carrier 計數由原82錨點的168增至210。
這些 C_B 見證有保留於 sign ledger，可以作為後續真實壓縮錨點，不能算作完整 PQCP。
完整閉合證明為 `results/L86_phase_million_coarse_radius1_closed_20261008/certificate.json`。

舊抽樣器的 checkpoint 會重新讀正在編輯的 script 檔案 hash，未必是已載入記憶體的 code hash。
保留原始一千萬 report，另外寫入 `runtime_source_prefix_audit.json` 說明差異。
4096 個 prefix trials 與真正執行舊來源快照的重播完全一致；這不是完整一千萬重播。

往後抽樣器直接重新啟動於凍結 source snapshot；worker imports 與 native builds 都使用該來源。
script digest 在開始時固定，使用共享 POSIX monotonic clock 計入 archive/setup/child imports。
主程序只證明一次內容界限，再把同一個合法 contents tuple 與順序由受控 process initializer 交給 workers。
不接受使用者 JSON 作為新的 UNSAT 證明。
已在產生 contents 時套用的 count-only predicate，不在每個 trial 重做：
它只取決於 target 與 contents，不取決於被抽出的支撐，所以不改變任何 trial 的判定。
4096 筆舊 frozen runner 與新共用父程序內容清單的輸入、輸出、工作量與 SHA 逐行相同。

## 6. IDE Run

`experiments/run_l86_norm_search.py` 現在預設開啟相位計數、projection groups 與共享 C completion。
按 Run 即執行一百萬個新支撐組；每組保留一或兩個完整目標，與原單一目標 trial 數目不同。
`RUN_TRIALS` 調整組數，`RUN_SEED=None` 每次產生並記錄新種子；指定整數可重播。
每次都輸出獨立 timestamp 目錄、全部 gzip trace、來源及 input/certificate 記錄。
`main.py` 的既有 Torch 流程沒有被替換；上述檔案是這條新數学流程的 Run 入口。

以該新流程另啟動二千萬組實際搜尋，seed 161803398：
`results/L86_projection_groups_twenty_million_20261008`。
只以其 terminal report 與 full verifier 計入完整新類別；未結束工作不算完成。
