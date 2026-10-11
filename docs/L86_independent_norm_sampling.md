# 直接在必要支持條件內搜尋

2026-10-07。這是新的流程實驗，不是把既有 CRT 必要條件改稱新定理。
`experiments/search_norm_supports.py` 直接產生合格支持形狀，再呼叫既有精確 C/G completion。
未從完整 PQCP inventories 取種子，也沒有把近解的低能量當成解。

## 數學條件及候選覆蓋

令 U、V 是兩列 C 的 active-support polynomials。在 F2[X]/(X^43+1) 內，
任何完整目標都必須滿足

\[
UU^*+VV^*=1+X^k+X^{-k}.
\]

其中 * 是 cyclic reversal，C 的 ±1 在模 2 後都是 active bit。
G 的 supports 為各列的補集；兩列補集 norm 中增加的兩個 all-ones terms 相消，
所以 G 同時滿足同一個必要 parity equation。這只解支持條件，尚未解任何完整符號或 PACF。

既有 `parity_support_family` 用完整 CRT/Frobenius norm parametrization 取得固定 U 的全部 V。
新 `cyclic_support_sampling.py` 先精確數出指定 weight 的 literal preimages，
再由獨立 PRNG 排程選 index，完整 phase product 取出該 literal mask。
不以 nearest-distance winner 代替其他 homometric preimages。
沒有 weight match 時，只證明固定 U 與該 weight 不相容，並非整个 target 不存在。
全部被選取的 V 都重算 integer weight、support parity signature 及 origin parity。

選 U 的 weight 及兩列 negative/zero/positive counts，逐一排程目前已驗證的 53 個 content choices。
每個 U 在其固定 weight 中生成；V 是該 U 的 weight-filtered norm family 裡的一個 literal preimage。
這不是全域 uniform support-pair sampling，也不是有限 trials 已覆蓋所有 supports 的聲明。
選擇不同 U、contents 或 phase 的排程是 heuristic；必要條件與每個 completion 才是精確數學約束。

## 完整 completion 與 UNKNOWN

兩列固定 supports 後，C 和 G 的精確符號問題獨立求解。
任一完整 UNSAT 排除這個 literal support pair；兩個 witnesses 才能重建完整 binary words。
重建需通過 independent full verifier、指定 target 及等價類去重才寫入 isolated inventory。
有預算截斷時保留 UNKNOWN，不計作 UNSAT，也不能宣稱已排除那個 pair。

Native 中新增的 `pqcp_weighted_support_at` 只作整數 weight-filtered index selection。
小長度 3、5、11 的測試逐一比較所有 index 與 literal CRT enumeration 的每個 preimage。
43 的測試另檢查 exact signature、weight、index 重播一致性。
權重在某些 family 中還受 mod 4 限制，因此測試與實驗以實際完整 weight count 為準，不猜測所有奇／偶 weights 都存在。

多 CPU workers 各自求解獨立固定支持問題；輸出按 trial index 記錄，便於完整重播。
`literal_trials.jsonl` 保留每個 U、V、contents、target、solver budget 結果與必要統計；
`report.json` 的 complete 只表示要求的 trials 全部處理完，絕非全域窮舉。

512-trial pilot：508 個完整支持 UNSAT、4 個固定 U 的 weight family 為空，沒有新解或 UNKNOWN。
後續 100,000-trial long run 的正式结果以 `results/L86_independent_norm_long_20261007/report.json` 為準。
這項搜尋的 throughput、已排除 supports 數目均不等同於「完整新解發現速度提升 20%」。

100,000 次已全部完成，約 90.36 秒：99,115 個完整支持 UNSAT，885 個固定 U 的 weight family 為空，
沒有 UNKNOWN，也沒有完整新解。來源已凍結在該目錄的 source_snapshot。
其中 9 組得到精確 C witness，再由完整 G sign enumeration 判定不可提升；
這 9 組不能算 PQCP。後續應研究這些 C witnesses 的支持／符號結構，避免只增加抽樣數。
後續更大的實驗把 process queue 限制在 2×workers 個 32-trial blocks，避免一次建立所有 futures；
trial seed、contents 與 phase index 的排程不變。每個 support 的 proof scope 仍限於其固定 literal 支持。

9 個 C witnesses 已另外作完整兩列二次交換鄰域及相鄰 content trades 檢查，
均沒有其他 coarse witnesses；詳見 [支持對稱剪枝與局部 trade](L86_support_orbit_pruning.md)。
