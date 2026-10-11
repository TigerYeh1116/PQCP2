# 單列 norm 的批次剪枝與見證域重用

2026-10-08。這兩項最佳化用於 `coarse_row_completion`：固定完整 signed C_A，
G_A 與整列 binary B 仍自由。目標是加快已指定搜尋域的完整求解，
目前配對實驗均沒有完整新 L86 解，因此不能視為已達成新解發現速度提升 20%。

## 完整低維符號纖維的批次求解

給定 C_B 支撐 $S$、negative subset $M\subseteq S$，有 $C_B=S-2M$。
令 $I_S(u)$ 為 active edge 數，$D_M(u)$ 為其兩端符號不同的 edge 數，則

\[
N_{C_B}(u)=I_S(u)-2D_M(u).
\]

固定 signed norm $\rho$ 後，模 4 給出必要線性等式

\[
\sum_{i\in S}\bigl(S_{i-u}+S_{i+u}\bigr)M_i
\equiv (I_S(u)-\rho(u))/2\pmod2.
\]

再加入 $\sum_iM_i\equiv n_-\pmod2$。
Gaussian elimination 保留全部自由變數，得到 $M=M_0+\sum_jz_jT_j$。
強制變數的 negative-count 上下界也只能作必要排除。

若自由維度 $q\le12$，且完整 $2^q$ 空間不超過本來的 sign budget，
批次後端枚舉 **全部** Gray-code assignments，檢查 exact negative count 及全部整數 norm。
首個真正的 signed witness 可交給新的 G 求解；全部檢查完且無見證才是 C_B-UNSAT。
大維度支撐標記為 deferred，交回原後端；deferred 不是 UNSAT，也不會漏掉原有 UNKNOWN。

所有新後端 UNSAT 分類，都在 native 內用另一條既有實作核對：

- range、GF(2) 或 forced-count 排除，要求原始 `pqcp_affine_sign_plan` 同樣排除；
- 小維度完整符號排除，再用原始 `pqcp_ternary_signs` 完整檢查所有 $2^q$ assignments；
- 發現不一致即失敗，不把該支撐裁掉。

SAT 見證再經 Python 獨立檢查所有整數 PACF、weight、negative count 及 literal support。
因此收益來自較少 Python/native 呼叫、集中處理與低維預處理，
不是跳過完整判斷或省略不利候選。

每批 256 個支撐，保存 packed classification records、schema、byte order 與 SHA。
同一批內可立即使用 C_B witness，避免先掃完整個幾百萬支撐族才開始提升。
提早找到完整 PQCP 時，只回傳第一個 model，沒有宣稱全部 C_B 支撐已閉合。

程式：`csrc/unary_norm_batch.c`、`solver/unary_norm_batch.py`。
選項 `--unary-batch-free 12` 啟用，`--unary-batch-free -1` 回退。
IDE coarse-witness 入口已使用 12；通用求解器及其他入口可保留回退配置。

## C_B 見證域快取

對完全相同的 $(\rho_B,n_-)$，C_B 的完整 signed norm 問題相同。
不同 target 的 G norm 可能不同，所以不能重用某個 G-UNSAT 判定來排除另一 target。
新的快取只保存 C_B 的 domain：

\[
\{S:\text{已有 C_B signed witness，或 C_B 尚未完整決定}\}.
\]

只有已完整證明 C_B-UNSAT 的支撐才可省略。
已知見證仍逐項核對 norm/counts/support；未完成的 C_B 支撐保存 `None`，下次重新求解。
僅因 G-UNSAT 而跳過的支撐仍須保留為 C_B 候選。
G 的 profile、negative counts 與所有 sign 解，每個新 target 都重新求解。

此方式也保存 UNKNOWN。完整 C_B 掃描尚未結束、因首個完整 SAT 提早退出、
或候選數超過 slot budget 時，都不能保存假裝完整的 domain。
快取只在目前進程建立，不匯入外部 JSON 作為 UNSAT 證明。
LRU 總量以 **支撐 slots** 限制，預設實驗上限 4096，不以可能很大的 profile 個數估計記憶體。

程式：`solver/unary_norm_carrier_cache.py`。
`--norm-carrier-cache 4096` 啟用，`--norm-carrier-cache 0` 回退。
它與原本只保存「整個 C_B norm 域為空」的 `CompleteUnaryNormUnsatCache` 是不同功能。

## 實際對照

兩個獨立配對都使用相同的 9 組實際 C 見證、18 個 signed C_A、全部 1512 個 targets，
各自凍結來源、交替順序、保留完整 ledgers。三次正式重複均完成，兩邊完整新解與 UNKNOWN 均為 0。

| 唯一新增項目 | 原流程中位數 | 新流程中位數 | 同一搜尋域耗時下降 |
|---|---:|---:|---:|
| C_B domain carrier cache | 45.9977 秒 | 40.9325 秒 | 約 11.0% |
| independently audited unary batch | 42.4729 秒 | 26.4249 秒 | 約 37.8% |

原始報告分別是 `results/L86_norm_carrier_cache_pipeline_paired_20261008/report.json` 與
`results/L86_unary_batch_pipeline_paired_20261008/report.json`。
後者另有 `independent_completed_children_audit.json`，重新核對全部 8 個已終止 children 的
錨點、targets、totals、來源 SHA 與 inventories。
不同實驗的回退時間不能混成單一比較，也不能直接相加兩項收益。

兩项同時使用的新 41 組 C 見證實驗，完整處理 82 錨點與 6888 targets，無 UNKNOWN 或完整新解。
這驗證了實際較大輸入上的完整閉合；它沒有舊版本同一輸入的時間配對，故不由此聲稱額外加速比例。

舊 9 組 C 見證的 radius-two 搜尋已完整結束：38,145 錨點、3,204,180 targets、UNKNOWN=0、完整新解=0。
證明存於 `results/L86_actual_coarse_witnesses_radius2_complete_audit_20261008/certificate.json`。
IDE 舊入口現在優先識別這份完整證明，避免再做已排除的 radius-two 區域。
要進行新的獨立抽樣，可按 Run 執行 `experiments/run_l86_norm_search.py`。

計數器 `assignments` 在不同後端包含的邏輯覆蓋與獨立核對工作可能不同，
不能把這些總數全部解讀為實際完成整數 PACF 的次數。
完整性應由域覆蓋、UNSAT/UNKNOWN 分類、原始 ledger 與 verifier 核對，而非單一計數器判斷。
