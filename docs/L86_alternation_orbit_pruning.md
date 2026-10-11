# 搜尋前的交替換號目標等價刪重

這是可用於一般 PQCP 等價類搜尋的精確對稱商，
不是相同 O、反射或特定構造的限制。
目前尚未證明完整 L86 新解發現速度提升 20%，主程式預設仍為關閉。

## 1. 非零 PQCP 目標必須重新推導

[Lumsden、Kotsireas、Bright 的原始論文 §2.1](https://cs.uwaterloo.ca/~cbright/reports/periodic-golay-pairs.pdf)
列出 periodic Golay pairs 的 simultaneous alternating negation。
Golay 的零目標在此作用下不變；本專案具有非零旁瓣，所以不能直接聲稱固定 k、η 不變。

令 L 為偶數，x'_i=(-1)^i x_i、y'_i=(-1)^i y_i。
因為周期跨越 L 不改變符號 parity，逐項乘積給

\[
T'(u)=(-1)^u T(u).
\]

所以例外位移 k 不變，η' = (-1)^k η。
每列 parity counts 由 (e,o) 變成 (e,L/2-o)。
再使用既有的整列換號、整列平移及列交换，就可選 content gauge 的規範代表。
交替換號正規化這些 content gauges：與一格平移交換只差整列換號，
因此對規範內容的作用仍是 involution。

## 2. L86 的完整目標商

| 原規範目標 | 原規範四個 parity counts | 交替換號後的規範目標 |
|---|---|---|
| k=1, η=-1 | (16,22,18,21) | k=1, η=+1, (16,21,18,22) |
| k=1, η=+1 | (16,21,18,22) | k=1, η=-1, (16,22,18,21) |
| k=2, η=-1 | (17,21,17,22) | 同一目標及內容 |
| k=2, η=+1 | (17,20,17,23) | 同一目標及內容 |

既有完整解發現 accounting 已把 simultaneous alternating negation 商掉；
搜尋前以前仍保留兩個奇數位移符號目標。
新刪重保留 3 個規範目標，既有嚴格 content bounds 下保留 40 組 paired contents，
原來是 4 個目標、53 組 contents。
其中兩個奇數目標各有 13 組內容，由 C/G 交換一一對應，專項測試核對完整集合。

函數只在「對應目標確實出現在輸入 pool」時刪除一個。
若只要求單一 η 或單一 profile，該 profile 必須保留，不能擅自改題目。
本商以新等價類／存在性為目的；若要列出每個不同 η 的 literal pair，
必須用反向換號恢復代表，不能把代表清單當成所有原始序列。

## 3. 為什麼族數不是加速證據

40/53 是參數族數量比例，不是相同工作量的實測時間比。
在隨機搜尋中，原兩個等價目標可以抽到不同的 literal supports；
它們不一定在重複執行完全相同的軌跡。
刪除一個目標並均勻分配給其餘目標，也會改變奇數／偶數目標的計算時間占比。
若新解集中在奇數目標，這種分配可能降低發現率。
因此不能宣稱「已提升 32.5%」，更不能用族數代替新解發現速度提升 20%。

## 4. 實作、相容性與測試

`solver/alternation_orbits.py` 建立規範 involution 與每個原 profile 的代表映射。
`TorchSearchConfig.alternation_orbit_pruning` 預設 False；
`TorchHybridConfig.continuous_alternation_orbit_pruning` 同樣預設 False。
啟用時需要既有 content symmetry pruning；continuous 與 direct compression 都使用減少後的 pool。
CLI 與 rollback 已接上兩條 production 路徑。
旧 checkpoint 沒有此欄位時視為 False，保持原來的目標 pool 及重播軌跡。

專項測試涵蓋所有 L6 原始 binary pairs 的換號 PACF 恆等式，
多個長度的完整 profile involution、單一目標不可刪除，
L86 全部 content gauges 的 literal PACF preservation、兩個奇數族全部保留 contents 的映射，
以及真正的 continuous/hybrid 設定能到達 3-profile pool。
整合測試目前 90 passed；這不證明新解發現加速。

另外延伸已有的「完整 UNSAT 支撐」快取：對這四個 L86 規範 profile，
交替換號把 C/G 支撐互換，C' 的負號数為 e-n_C。
奇數目標不需其他 row gauge；偶數目標交换兩列後回到同一規範內容。
所以相反奇數符號或偶數的 complement-plus-row-swap 支撐，也可重用同一完整無解證明。
`complete_pqcp_supports(...,orbit_pruning=True,modulation_orbit_pruning=True)` 是獨立回退選項，
預設 False。非這四個 profile 則退回原來的 dihedral key。
UNKNOWN、SAT 都不能寫入排除快取；來源 masks、target、負號數均隨命中回傳，避免省略變換前提。
測試對全部 53 組 contents 直接重建原長度 binary rows，
核對 C/G 交換、PACF 換號及 key 相同；另以實際 L43 UNSAT 與 UNKNOWN 記錄核對來源及回退。
這一項尚未有實際發現速度證據，也沒有混入已凍結的 target-pool 長測。

## 5. 正在執行的完整發現驗證

`experiments/run_alternation_discovery_validation.py` 可在 IDE 直接按執行。
預設先從 8 個 CPU workers 完成 1,000,000 次独立 norm-compatible 支撐搜尋，
每組都求解完整 C/G signs；再循序執行 3 seeds×rollback/candidate，
各 600 秒、MPS batch 16,384 的原 production pipeline。
來源與 proof records 先凍結，子程序從凍結目錄執行；兩階段不並行，避免計時互相干擾。
所有完整發現都扣除初始 inventory，配對階段也扣除前面 norm 階段已找到的解。
只有独立 full verifier 通過的 L86 等價類才算發現；coarse SAT 不算。

本次資料位於 `results/L86_alternation_discovery_long_20261007/`。
進行中／失敗必須以 manager、norm 及 paired report 的當前狀態為準。
兩個配對 variant 都沒有新解時，只能報未量得發現速度，不能宣布達成 20%。

百萬次 norm 階段已完整完成：989,744 組 sampled support pairs 完整 UNSAT，
10,256 組固定 U 的 V family 沒有指定重量元素；無 UNKNOWN、無完整新解。
報告 elapsed=806.286536 秒。這只是這批樣本的結果，不是 L86 不存在的證明。
此階段有少量其他驗證工作同步執行，也不是與 rollback 的配對時間比较。
後續 GPU 配對階段會獨立循序執行；目前尚未完成六次配對，不能報新解速度。

後續新增整合測試 38 passed，含新 flag checkpoint 的重播，以及
舊 checkpoint 沒有欄位時仍保留四個目標、逐步參數相同。
