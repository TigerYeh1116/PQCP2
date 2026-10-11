# 新解發現目標的驗證狀態

目標為完整且新的 L86 等價類之發現速度提升至少 20%。目前尚未達成。
候選、coarse SAT、較低能量、支持數減少或單一階段加速，均不代替完整新解。

## 2026-10-08最新階段

目標仍未達成；root `86.txt`仍不存在，所有本輪completed experiments的完整新解均零。
新增方法及證據如下，早先段落保留各自當時快照，不以新預設覆寫舊實驗。

| 完整實驗 | 嚴格數學條件／範圍 | 結果 |
| --- | --- | --- |
| joint Gram三次配對 | rational outward intervals，48,648個相同fixed triples／輪 | local時間減少15.77%；18,972證書獨立核對；fullnew=0 |
| single norm的p² audit | PSD通過的2,585個distinct profiles，不能用於兩unknown norms之和 | 823排除，31.84%；所有5個positive-periodic-model profiles保留；不是發現率 |
| finite norm／trace三次配對 | F_(257²)共軛根，nonzero square discriminant排除，zero fallback保留 | local時間減少13.98%；18,975證書獨立核對；fullnew=0 |
| sparse PSD三次配對 | 稀疏目標delta共用dot；證書與逐一PSD完全相同 | local時間減少12.92%；37,950 finite證書獨立核對；fullnew=0 |
| 全部targets的affine finite先行 | 同一必要條件改變順序，全部full輸出與targets保留 | local慢39.84%；3,083,799證書獨立核對；fullnew=0；未預設使用 |
| even-only affine finite先行 | odd groups仍保留單profile PSD先行 | local慢32.13%；1,558,128證書獨立核對；fullnew=0；未預設使用 |
| 最新八centers，兩半列radius2／3 | 一個bounded varied半列，另一個43bits全部自由，全12roles | 48,648／386,744 eligible fibers；皆完整fullnew=0 |
| 最新八centers，兩半列radius4 | pinned兩個closed parent centers＋六個完整parallel children | 13,123,008原始triples，5,530,424 eligible fibers；全八中心完整fullnew=0 |
| 最新八centers，三半列total radius2 | 兩個nonzero varied masks共享budget，第三列自由，僅一列固定 | 177,504原始triples、32,600 eligible fibers；完整fullnew=0 |
| 最新八centers，三半列total radius3 | 同上，更大完整區域；frozen舊版全部finite先行 | 7,632,672原始triples、1,397,480 eligible fibers；完整fullnew=0 |
| 最新八centers，四半列total radius3 | 三個nonzero varied masks共享budget，第四列自由；所有四個roles | 2,544,224原始triples、667,040 eligible fibers；完整fullnew=0 |

joint Gram／finite percentages是各自單獨配對，不可加總。
manager時間不含pinned parent原始工作，不能當作同一版本配對benchmark。
3半列radius3的舊快照有29,340,245 finite-target排除、13個PSD排除，沒有norm visits；
long run保存全部證書，但沒有聲稱已將這29M證書全部逐一做第二次Python離線audit。
三次配對的證書則全部在計時之外獨立核對，與長測的保存範圍分開記錄。

數學推導：[Gram](L86_joint_gram_pruning.md)、[single norm p²](L86_prime_norm_obstruction.md)、
[finite norm／trace](L86_finite_field_trace_pruning.md)、[affine batching](L86_affine_profile_batches.md)、
[平行區域證據](L86_parallel_region_evidence.md)、[三／四半列完整region](L86_three_half_region.md)。
新增文獻適用性與自己的推導另記於 [文獻更新](L86_literature_update_20261007.md)。

目前IDE研究入口預設PSD先行，開啟batched PSD、finite與Gram；affine finite先行預設False。
core保留各方法的False回退；`main.py`的原始MPS流程沒有改成此半列補全實驗。
新的四半列搜尋入口是 `experiments/complete_four_l86_halves.py`，可直接按IDE執行鍵。
其total radius3涵蓋四個free roles；三個known rows各有至少一個flip，第四列全部43bits自由。
`results/L86_four_halves_total3_top8_20261008/`已完整完成。
接著四半列total radius4的四程序長測已啟動於
`results/L86_four_halves_total4_parallel_20261008/`，預定總triples162,830,336；
只有全child完成才能把本次report標示complete，沒有混入舊r=3的完成計數。

affine全目標版本的整合專項74 passed，後續even-only及finite專項9 passed；
另最新四半列及平行manager／實驗核對7 passed。
全專案pytest收集遇到兩個既有缺少的modules：`benchmark_mps_acceleration`及`benchmark_torch_mathematical_loss`。
排除相應兩個test files後為1,465 passed、14 skipped；環境另有NumPy未安裝的PyTorch warning。
沒有把這次部分收集的回歸稱成「整個專案所有測試全通過」。
新增code的py_compile與git diff --check通過。

目前最有價值的嚴格結論：對八個選定centers，若解只改兩個半列，兩列距離都至少5；
若只改三個半列，兩個最小非零距離之和至少4（total radius3子區域已完整排除）。
這只適用固定input附近，不是一般L86不存在證明。
下一步讓四個半列一起變化，接受者維持全長整數verifier；沒有將低能量、有限域原料或零對零計時當成新解。

## 完整 pipeline 的六次配對

`results/L86_discovery_paired_20261007/report.json` 已完成。
三個 seeds 123/456/789，各比較 observation projection 關閉／開啟，預算各 300 秒。
六次皆沒有 verified new class，也沒有 exact compressed match。
該快照中的投影增加 CPU 成本，candidate 端 formation steps 少約四分之一。
因此不能宣稱此完整流程較快。
比較的 CLI options 沒有逐一涵蓋 IDE 使用者所有互動選項，亦不是整個專案最初版本的比較。

## 精確投影 cache 的固定工作量比較

後續加入 bounded 512-entry memoization，cache 只保存純整數投影結果。
相同 16384 lanes×2000 steps、五次交錯重播：
未 cache 中位 33.66710 秒，cache 31.79705 秒，時間減少約 5.55%。
候選、archives、scores、Adam theta 及 work counters 一致；完整新解皆為零。
記錄 `results/L86_projection_cache_ide_20261007/report.json`。
這不是 20% 新解加速證據，亦沒有混進前述固定快照的六次結果。

## 本輪新增構造與局部精確搜尋

[數學推導及範圍](L86_interleave_derivation_20261007.md) 記錄 GS/D-optimal 原料、
有限 catalog 擴大、radius-4、4+2/2+4 與 4+4 的必要條件配對。
產生 full energy 8 的新候選來源，但尚無完整新解。
GS 長測與 direct-source-color transfer 比較的結果應以各 isolated report 的 complete 欄位為準；
尚在執行或有 UNKNOWN 的工作不能被描述為完整排除。

目前主程式維持原有已驗證的數學剪枝與回退路徑。
本輪研究分支先接到獨立的既有修復實验；只有完整發現率的比較能支撐後續預設變更。

## 直接必要支持族的完整 long run

新增 [直接 norm 支持抽樣](L86_independent_norm_sampling.md)，100,000 trials 已完成，
9 組 exact C witnesses 皆由 G 完整判定不可提升，沒有 UNKNOWN、没有完整新解。
後續 [dihedral UNSAT 剪枝與 C trades](L86_support_orbit_pruning.md) 是可回退的實驗選項；
目前這批資料没有對稱快取命中，也没有新的局部 C witness。
一般回歸最後一次為 1,336 passed、14 skipped；之後新增的 orbit/trade 實作另以專項測試檢查。

## 整數核構造與更大區塊的後續階段

新增 [odd-kernel 整數格推導](L86_odd_kernel_lattice.md)：
相同 perfect O 的 168 個子問題、相同 one-flip O 的 1,764 個子問題皆完整 UNSAT。
後者保留整數右逆證書 B R=I 與全部原料商映射。
只是有限構造族排除，不能用來限制一般 PQCP 搜尋。

[支撐區塊](L86_exact_support_blocks.md) 從 9 組長測 C witnesses 出發，
72 個 m=16 區塊的 16,361 組必要配對全部完成 signs 求解，無 UNKNOWN。
找到一個新訪問的 exact C support，但 G 完整 UNSAT，完整新解仍為零。
此方法只固定區塊外支撐，全部符號自由，已跨過原來局部符號交換的限制。

[四色整數區塊](L86_integer_joint_blocks.md) 修正 MPS 浮點及張量後端的能量核對問題，
改用直接 Metal 整數核。完整 3,763,200 分數矩陣與獨立整數 oracle 相符。
6 個較長 chains 共 1,920 個完整區塊未見最小配對 audit error，
但最小 full energy 仍為 8，完整新解為零。
失敗的舊 backend 執行保留未完成狀態，不能當作排除或加速證據。

將支撐區塊擴大到 m=20 後，4 個完整區塊的 199,911 組必要配對全完成判定，
又走到 19 個未訪問 C supports，仍沒有完整新解。
另推導 [最小差分誤差 shell](L86_minimum_difference_error_shell.md)，
原來 9 組及新 bank 20 組支撐的全部 12,180 個 energy-4 profiles 都完整 UNSAT。
這些固定支撐在對應 contents 下 full energy≥8，不能提供更低能量的起點。
不能把這個固定支撐下界誤作一般 L86 不存在或整個搜尋空間的下界。

另外證明 unsigned support parity 有 q 個目標缺口時，full energy≥q，與 signs 無關。
24 個 GS specimens 皆 q=energy=8，已完整核對，故固定這些支撐的符號最佳化無法改善。
正在進行 GS 周邊 m=18 的支撐區塊搜尋；其全符號求解進度必須以 isolated report 為準。

GS 周邊 m=18 已完成 96 個區塊、333,395 組完整 signs 判定，無 UNKNOWN、無完整新解。
下一個一般搜尋剪枝是 [交替換號目標商](L86_alternation_orbit_pruning.md)：
4 個規範目標變為 3 個，53 組 contents 變為 40 個，
但這只證明等價代表仍有覆蓋，不能用族數比例代替新解速度。
已啟動凍結來源的百萬次完整搜尋，後接六次各 600 秒的完整 MPS 配對驗證。
主程式預設仍關閉此新商，旧 checkpoint 的軌跡維持原設定。

該百萬次完整搜尋已完成，989,744 組 sampled support UNSAT、10,256 組固定側重量族空，
沒有 UNKNOWN 或完整新解；這不是全域不存在證明，也不是配對速度比較。
manager 已進入六次各 600 秒的完整 MPS 配對階段，尚未完成，目標仍未達成。

上述 manager 現已完成：三個 seeds 各 rollback/candidate 共六次 600 秒配對，
兩側完整新等價類均為 0，六次正常結束，沒有 timeout。
總記錄 `results/L86_alternation_discovery_long_20261007/report.json` complete=true，
elapsed=4450.019273417 秒。不能用 0 對 0 計算提升率，20% 目標仍未達成。

配對工作進行時另推導 [自由短區塊的模 8 線性投影](L86_boundary_mod8_projection.md)，
新核心保留任意前綴的完整 baseline，允许跨 parity-content families 的短窗口修復。
原有十個測例及新增 L86 非空投影測例均已通過；
兩個原始 periodic82 前綴各自所有 256 個 tails 已判定無解。
完整 periodic82 orbit 与 GS window 長測在配對計時結束後已執行，均完整無解。
前者涵蓋 268,960 個前綴對／68,853,760 個 raw tail assignments；
後者涵蓋六個非解中心的完整四位元窗口。
目前既沒有完整 L86 新解，也沒有達到新解速度提升 20% 的證據。

接著完成 [正號奇數仿射格構造](L86_perfect_affine_kernel.md)：
168 個完整 fibers、1,344 個 c 固定支撐子問題均 UNSAT，沒有 UNKNOWN。
再加入 [內容自由、任意位置的翻轉配對](L86_unrestricted_flip_join.md)，
完整決定六個 GS 中心、十個 joint-chain 中心及最新長測 log 的十六個中心之
radius-4 per-row product balls，仍沒有完整新解。
radius-5 擴大搜尋也已完成：十個中心全部 UNSAT，11,180 個模 8 必要匹配
均判定不符合完整目標，沒有 UNKNOWN，elapsed=402.787094166 秒。
記錄 `results/L86_joint_unrestricted_radius5_20261007/report.json`。
新修復入口在 IDE 執行時預設讀取最新根目錄長測 log，主程式預設仍維持原流程。
本輪整合專項測試 26 passed；編譯檢查與 git diff --check 通過。
沒有找到完整新解，也沒有達到新解發現速度提升 20% 的證據。

## 後續的精確索引剪枝及新原料

DC/ALT 提前到模 8 索引已完成十八個固定工作量配對，
保持覆蓋範圍與完整 PACF 檢查數，查詢數恰減半，局部中位時間减少 46.5536%。
兩側完整新等價類均為零，這不是新解發現速度或整個 pipeline 的改善證據。
最新長測十六個中心已再完成 radius-5，全部 UNSAT。

[SDS norm 原料擴展](L86_sds_norm_expansion.md) 完成 42,321 個 D-optimal 固定側子問題，
得到八個原料等價類，其中三個不在本專案先前完整 catalog。
完整 odd joins 及八個最佳 seeds 的 radius-5 修復都没有完整新解。
接著放寬 individually-perfect O，完整決定 9,330 個 Legendre 固定側問題，
六個 O 等價類中兩個 non-perfect，與 336 個封閉 E 原料交叉搜尋 3,360 個構造族。
仍沒有完整新解，沒有 UNKNOWN，最低非解 seed energy 為 36。
記錄 `results/L86_legendre43_norm_interleave_trades2_20261007/report.json`。

本輪整合專項測試為 68 passed；包括原 cyclic CRT 搜尋／抽樣、原 interleave、
新完整 SDS 補全、native odd vectors、內容分組與舊邊界／仿射格。
20% 完整新解目標仍未達成；下一步可放寬為四列 summed PACF 為零的完整固定三側補全。

## 四列補全、任意43位元半列及 exact norm／trace

上述四列補全現已完成實作。第一波四個free rows共有158,829個固定三側子問題，
得到126個原料代表；第二波549,665個子問題再增加60個至186類。
全部原料的完整odd joins都没有完整新解，最低非解能量仍8。
詳見 [SDS／PCS4原料擴展](L86_sds_norm_expansion.md)。

[任意半列補全](L86_half_row_completion.md) 決定最新59個log中心、236個fibers，
各未知半列允许全部2⁴³位元及所有必要重量／target choices。
全部完整新解零；這不等於全域不存在，也不是重新測試同一個小翻轉半徑。
另由DC／ALT原始式推導全部四個parity sum形狀，並核對所有signed integer可能性。

[有理PSD證書](L86_certified_profile_psd.md) 在65個中心、260fibers的三次配對中，
全部periodic preimages／phases／完整模型一致，局部中位時間减少34.4644%。
完整新解均零，尚未達成完整新解發現速度20%目標。

[norm／trace完整求根](L86_norm_trace_pruning.md) 將一般nonzero-partner分量
由129個norm phases縮到至多兩根，generic全CRT至多八個literal masks；
zero partner完整回退，未知位元weight／integer PACF及full86 verifier仍必須核對。
Python配對慢1.7116%；加native field、共用中心準備及literal rotation audits後，
最新五次配對中位時間2.009983541→2.002979750秒，局部减少0.34845%。
所以trace仍預設關閉，主程式的原始預設沒有加入這個研究分支。
三次實作均有獨立凍結來源的配對結果目錄，沒有覆寫旧結果。

[兩點impulse／任意skew O](L86_impulse_skew_O.md) 的122,880個c及全部skew norm roots，
已完整決定無解，沒有perfect-O限制、沒有UNKNOWN。
進一步放寬為 [四點g／任意O](L86_same_O_weight4.md)，
預設全5,740個reflection representatives、對應11,480個normalized literal g。
模4的完整affine C、integer weight、有理PSD及anchored norm／trace在native完整枚舉；
這個新族的完整實驗記錄將另行更新。

目前整合專項71項通過；四點g新增專項與norm／trace合計8項通過，
包含完整小長度affine空間及一個L43完整524,288-assignment fiber的逐階段native／Python計數。
後續整合檢查與全族結果以下方更新為準。
至此仍沒有完整L86新等價類或「新解發現速度提升20%」證據。

四點g全族現已完整結束，5,740個代表、1,504,706,560個affine assignments，
完整新解零，elapsed=176.217270625秒。
其中186個literal C/O通過全部odd equations，但未通過even equations。
已完整重播177個非空fibers，原始計數全部相同，取出186個非解等價類，
最低精確能量160；這沒有改善先前最低非解能量8，也不是新解發現。
全部odd survivors的獨立原始PACF與native energy核對通過；
接著對新中心補全四個任意半列，放寬同O及g重量限制，記錄另存。

上述186個中心的744個任意半列fibers也已完整決定，全部無解、沒有UNKNOWN，
`L86_weight4_seeds_all_half_rows_20261007` elapsed=10.489983958秒。
整合專項76項通過，包含seed輸出buffer擴大重跑與獨立原始能量核對。

另外發現先前兩點impulse只是target與g位置一致的一族；
已自行推導 [全部21種g位置及唯一O對稱部分](L86_same_O_weight2_general.md)，
允許O不是skew，並完整訪問22,020,096個affine assignments。
2,580,480個C重量符合者及44,944個literal O已全部核對，
8,068個odd-integer符合者在even equations失敗，完整新解零、無UNKNOWN，
`L86_same_O_weight2_complete_20261008` elapsed=0.515736秒。
其odd survivors可供較一般的補全／導航，但不是完整新解。

兩點g的odd survivors也已完整重播核對：8,068個literal、4,034個非解等價類，
最低能量96，全部原始odd位移核對與native energy一致。
`L86_weight2_odd_seeds_complete_20261008` elapsed=36.01409175秒。
正在以四個任意43位元半列的完整補全放寬構造限制，結果另存。

該4,034個two-impulse中心的16,136個任意單半列fibers現已全部決定，
完整新解零、沒有UNKNOWN，`L86_weight2_seeds_all_half_rows_20261008`
elapsed=115.97766591599999秒。所有新增general two-impulse及共用native核心整合專項79項通過；
py_compile與git diff --check通過。
數學上下一個較合理的放寬是 [兩個半列可改、其中一列完整自由](L86_two_half_row_next_derivation.md)，
需保留全部有序角色與任意位置翻轉，不能延用固定三列UNSAT結論。
此兩半列擴大目前只是推導設計，尚未實作；不是已完成的剪枝或加速證據。
目前任務仍未達成：沒有完整L86新等價類，尚無新解發現速度提升20%的驗證。

這一階段一般回歸為 1,356 passed、14 skipped；
後續新增 shell 的專項測試另外涵蓋完整小 error cube 及原始 binary 空間的正例。

上述研究流程均有独立 results 目錄；主程式預設沒有加入相同 O 或反射限制。
文獻與本專案推導的區別記於 [文獻更新](L86_literature_update_20261007.md)。
目前尚未達成「新 L86 完整等價類發現速度提升至少 20%」。

## 2026-10-08：較大有限區域及壓縮列補全更新

先前兩半列擴張已實作並完整判定 root 最新長測八個中心的 radius4 區域：
13,123,008 指派、5,530,424 eligible fibers，新類零。
三半列 total radius3：7,632,672 指派、1,397,480 eligible fibers，新類零。
四半列 total radius3：2,544,224 指派、667,040 eligible fibers，新類零。
四半列 total radius4：162,830,336 指派、23,024,744 eligible fibers，新類零。
最後一項八個 child 完成后 coordinator 有 list／tuple 彙整錯誤；
獨立核對輸入、來源、全部計數及完整解庫後已 reconciliation，complete=True、elapsed=null。
不能把彙整失敗時間或不同來源的平行總時間當配對加速率。

固定完整 binary A、另一列全部 86 bits 自由的 radius0／1 已完整判定。
radius1 有 1,392 個 distinct anchors、116,928 個 targets、無 UNKNOWN、完整新類零。
這比只補一個半列更廣，但仍是指定 anchors 的條件性無解。

為放寬固定 A 的限制，已實作只固定 signed c_A、所有 g_A 與 binary B 自由的搜尋。
其投影區域為完整 c_A L1 球；單列 c_B 的全部模2支撐與整數符號保留，
G 方程獨立耦合求解，任何候選接受前通過原始全長 verifier。
root 長測八個中心 radius2 的原凍結版本仍在執行，
最近 checkpoint 為 16,512 anchors／1,387,008 targets、36 個壓縮支撐見證、完整新類零。
這是部分進度，不是整個 radius2 已排除；最新結果以實驗 report／ledger 為準。

新增單列範數 mod4 障礙與 mod8 全根／線性 kernel 補全，
另有只重用 C_B 完整空集合的 bounded cache，以及完整無解後才取商的支撐反射。
不同原始 κ 的 G 目標不互相混用，SINGLE norm 規則不誤用於兩個 G 範數的和。
完整流程配對已啟動於 `L86_coarse_pipeline_pruning_paired_20261008`，
必須等每份宣告區域完整結束才能比較；零／零新類仍不算新解發現速度。

又自行推導固定負號數的精確圖割投影，對單列與兩個未知範數之和都合法。
36 個實際固定 G 支撐分支完整配對結果：新排除零、無 UNKNOWN、無完整解，
中位數 0.076823→0.092456 秒，局部增加約 20.35%；暫不預設啟用。
已改成 affine 先行及較大分支才投影；完整 c_A 區域配對另外啟動，
不將這個短區域當成放棄方法的理由。

`experiments/complete_l86_coarse_witnesses.py` 提供新的 IDE 執行入口：
直接重新核對先前 9 組真實壓縮範數見證，使用兩邊所有 c_A 的完整 L1 radius1 球，
g_A／B 完整自由，不把壓縮見證冒充完整 PQCP。
已啟動於 `L86_actual_coarse_witnesses_radius1_20261008`，尚未完成。

新補全／cut／輸入專項 21 passed；上一輪可用一般回歸 1,465 passed、14 skipped。
全目錄 collection 有兩個既存缺失 benchmark modules，尚不能宣稱全 repository 測試全通過。
共用 driver 已修正 coarse UNKNOWN 統計；舊凍結執行結果仍須按原來源及分支 ledger 審核。
完整推導及排除作用範圍見 [壓縮列與 2-adic 補全](L86_coarse_row_and_two_adic_completion.md)。

目前完整新 L86 類仍為零，任務保持進行中，尚未達成新解發現速度提升至少 20%。

### 同日後續：閉合證据承接與新反射模型證明

原 root radius2 舊程序已正常中斷、gzip 正常關閉；独立核對 16,766 個完整 anchor 的
全部 1,408,344 targets（均完整 UNSAT，無 UNKNOWN），來源與所有 ledger SHA 已保存。
新程序在 `L86_fixed_coarse_rows_radius2_resumed_20261008` 承接這些已閉合結果，
剩餘工作用模4／模8、單列空集合快取與支撐反射處理；沒有時間窗重啟或丟掉前綴。

9 組真實壓縮見證的 radius1 區域已完整閉合：1,179 anchors／99,036 targets，
42 個 C_B 壓縮支撐見證、完整新類零、無 UNKNOWN，586.780746 秒。
逐項審核後 radius2 在 `L86_actual_coarse_witnesses_radius2_20261008` 承接所有閉合 literal anchors。
新 IDE 入口預設 radius2，無參數直接執行时自動承接已審核的 radius1 bank。

新模4／模8等方法的完整區域配對已完成兩組，同为 1,046 anchors／87,864 targets：
674.289→274.345 秒、722.424→272.028 秒；全部逐 target／anchor／source SHA 审核。
第三組以 report 為準；零／零新解不能換算為新解發現率。

圖割先於 affine 的另外三組 radius0 全流程配對也已完整結束：
18 anchors／1,512 targets，51.698→264.896 秒，約 5.124 倍耗時。
有 19 個 cut 排除，但全部已是 affine 空集合，所以没有額外剪枝收益。
保持可選而非預設；affine 先行的新配對仍完整執行。

另有一個完整數學新結果：兩條 C 或兩條 G signed rows 不能同時反射對稱。
模4條件化為 43 方程／20 變數、rank17 線性系統；
負號目標四個分配都不一致，正號目標兩個非空 affine 空間各僅8個支撐，全部不符必要重量20。
完整 parity矛盾與kernel／weight證書已保存，3項独立小長度／43點專項通過。
這只排除對稱 signed model，沒有以對稱支撐排除任意符號、也不是一般 PQCP86 不存在。

最新可用一般回歸 1,500 passed、14 skipped（另排除兩個既存缺失 benchmark modules），
最新本階段專項32 passed；IDE defaults 完整 smoke 2 anchors／168 targets 通過。
數學、資料、範圍與全長验收仍分開記錄，所有完整新類仍为零。

三次新範數全流程配對現已全部完成，中位數709.026→272.028秒，
相同完整區域判定時間減少61.63%；全部8個warm／repeat children的target、anchor與來源核對通過。
每次區域仍為1,046 anchors／87,864 targets，完整新類均零；new_discovery_time_reduction=null。

affine先行後的圖割三次完整配對也全部結束，51.297→52.373秒，增加2.10%，
沒有額外收益；較早版本的412%額外成本已修正，但保持圖割非預設。

同範數／支撐dihedral＋common-unit完整anchor證明重用也已完整審核四個全部區域，
1,046／33,907及1,179／38,145 literal anchors均沒有額外proof-domain重複。
保留數學推導與長度43非trivial homometric正控制、完整小長度存在性測試，未加入零命中預設快取。
最新本階段專項35 passed；沒有將條件性無解、剪枝量或速度下降／區域時間改善當成新類發現證據。

root八中心的完整c_A L1 radius2區域現已全閉合：33,907 anchors、2,848,188 targets，
62個C_B壓縮支撐見證、完整新類零、無UNKNOWN。
原始與續跑的所有target、anchor、輸入順序、來源SHA及總計已遞迴核對，final report一致。
證據另存 `L86_fixed_coarse_rows_radius2_complete_audit_20261008`，原始ledger全部保留。
任何完整解的C列與這16個來源投影的距離必須至少3；不作全域不存在或新發現率推論。
續跑時間2699.357秒不含舊前綴計算，不稱整個區域從零的時間。
9組壓縮見證的radius2另一区域仍在執行，預定38,145 anchors／3,204,180 targets，
只承接已審核的radius1，不因觀察時間窗中止；最終進度以report為準。
