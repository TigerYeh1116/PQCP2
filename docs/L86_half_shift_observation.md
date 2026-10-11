# L86：保留奇偶 content 的最小 Hamming 半週期投影

日期：2026-10-07。本次自行推導並實作於 `solver/half_shift_projection.py`。
它在候選觀測時補充接近原 binary pair 的必要條件可行候選；
原始候選仍受完整觀測，rank projection、continuous loss、Adam 更新公式皆維持原式。

## 1. 必要條件與精確內容

L=2d，d 為奇數，每個 antipodal pair (i,i+d) 有一個 even 與一個 odd 位置。
某列有 e 個 even ones、o 個 odd ones、n 個 11 pairs。
pair types 的個数完全決定為

N_11=n，N_10=e−n，N_01=o−n，N_00=d−e−o+n。

因此 max(0,e+o−d)≤n≤min(e,o)。
其 unequal pairs 數 z=e+o−2n，半週期 PACF 為 2d−4z。
PQCP 的非零位置是 k、L−k，不能包含自對稱的 d；故

\[
z_A+z_B=d,\qquad
n_A+n_B=(e_A+o_A+e_B+o_B-d)/2.
\]

L86 canonical profiles 的右側全部等於 17。
n 同時是 normalized compression 的負號數，所以可使用已驗證 coarse／binary content 範圍。
候選 n-pairs 使用既有 `search_paired_contents`，證據無法驗證時會保守回退。

## 2. 最少位元修改的證明

保持某列 e/o 個數，將 n 改成 n'，至少須修改 2|n'−n| bits。
例如 n 增加 Δ，至少要創造 Δ 個新的 11 pairs，
每個至少須一個 0→1 修改；固定 ones 數又要求相同數量的 1→0，合計至少 2Δ。
減少 n 的證明同理。

這個下界可以達到：
增加 n 時，在 10 pair 的 even 1 與 01 pair 的 even 0 間交換，得到 00 與 11；
減少 n 時，在 11 pair 的 even 1 與 00 pair 的 even 0 間交換，得到 01 與 10。
可行內容的 pair-type 個數保證有足夠的 donors／receivers。
每次交換恰改兩個位元，且保留 even/odd ones。

對兩列有限枚舉所有允許 (n'_A,n'_B)，最小化

\[
2(|n'_A-n_A|+|n'_B-n_B|).
\]

選到的距離為全體指定內容及半週期可行 binary pairs 的精確最小 Hamming distance。
同距離時固定選較小 n'_A，再以固定位置順序交換；没有使用經驗分數排除配置。
已符合所有條件的 binary pair 距離為零，保持原樣。

上述是可獨立回退的 plain 構造。新預設另外改善同距離的選擇：
枚舉全部具有相同最小距離的 n-pairs，在兩種 row 處理順序中，
每步檢查全部合法 pair-type swaps 的精確整數 target energy，選能得到最低值的交換。
所有交換都單調將 n 向指定 n' 移動，且不會再次修改先前已處理的 pair，
因此仍有同一個最小 Hamming distance 證明。
plain 構造亦保留在候選集合內，最終 energy 不會比 plain 差。
這個 greedy 次要準則**不保證**全體同距離 binary pairs 中的全域最低 energy。

`csrc/half_shift_projection.c` 精確計算兩位元翻轉的 PACF 差：
令翻轉位置 p≠q，原 signs x，對 u≠0，

\[
\Delta P(u)=-2x_p(x_{p+u}+x_{p-u})-2x_q(x_{q+u}+x_{q-u})
+4x_px_q([p+u=q]+[q+u=p]).
\]

ΔP(0)=0，所有位置 mod L。
由原殘差 R 與 ΔP 直接求新的 E_full+2E_2；不使用浮點近似排序。
Python 對選中的結果獨立重算完整 PACF、energy、內容、半週期与 Hamming distance。

**這不是對連續 logits 的最近 Euclidean 投影。**
不能替換 loss 中 `BatchedPQCP.project`，否則原 nearest-projection／gradient 推導不成立。
本實作只用於離散觀測後的補充候選。

## 3. 與現有流程的銜接與回退

新 L86 `TorchSearch` 預設啟用 `half_shift_observation_projection=True`。
原 top score／target energy 與全部 score-zero lanes 先照常處理，
再對這些原本選中的有限 lanes 產生有正距離的補充候選。
新增候選以 GPU 重算 PACF，再經 Python 完整整數 PACF、內容、energy 與 verifier 檢查；
進入原 bounded archive、C polish、completion 及去重流程。
沒有將投影成功、半週期零、低 energy 當成完整解。

額外處理數計入 `candidate_evaluations`，且 payload 保存
`candidate_construction=minimum_hamming_half_shift` 與 `projection_distance`。
補充候選不能影響 raw lane 的停滯／rebirth 判斷。
`half_shift_observation_projection=False` 回退原觀測方式；其他長度自動保持原方式。
`half_shift_projection_energy=False` 只回退到 plain 最小 Hamming 構造。
舊 checkpoint 沒有這個欄位時明確使用 False，保持原先觀測行為；新 checkpoint 保存選項。
IDE 新建 L86 搜尋自動啟用，不需要新增指令或提問。

## 4. 驗證與計時

`tests/test_half_shift_projection.py` 對 L6 的全部 4,096 input pairs，
直接枚舉同內容、half-shift-zero 的完整 binary set，比較真正最小 Hamming distance；
也限制指定 n-pair 檢查相同結論。對 L86 四個 profiles 的 512 組隨機 input，
驗證 content、半週期、已證明負號數範圍、精確最小距離及 idempotence。
`tests/test_torch_observation.py` 比較 enabled／disabled，
檢查原候選保留、完整 integer profiles、原 Adam／停滯軌跡相同及 checkpoint continuation。

第一輪 MPS、512 lanes、64 步、五次交錯：
原 eligible archive candidates=21，新=50；每秒此類候選 88.03→154.66。
formation 時間 0.23855→0.32330 秒，成本增加而候選數增加。
此 eligible 僅指符合半週期與證明內容，並非完整解。
記錄：`results/L86_half_shift_observation_20261007/report.json`。

第二輪 MPS、library 的 2048 lanes、2000 步、每 25 步觀測、五次交錯，
使用原一般觀測方式（fast_observation=False）：
原 eligible archive=76，新=187；中位 formation 10.2704→12.8564 秒。
每秒此類 archive candidates=7.3999→14.5453，增加约 96.6%。
兩側 Adam 最終參數逐位相同，最佳 score 都是 56，完整新解都為零。
記錄：`results/L86_half_shift_observation_production_20261007/report.json`。
2048 是 library 的 batch，並非本機 IDE 自動硬體 batch；也尚未量測實際 C seed export／completion 收益。
所有「eligible candidates/s」只是必要條件吞吐量，不能宣稱新 L86 解速度提升 20%。

第三輪以 IDE 的 16,384 lanes、2000 步、五次交錯，進一步核對實際 C seed export：
plain 构造的 eligible archive 74→177，但 eligible exported seeds 只從 42→44；
含 formation/export 的此類種子吞吐量 1.4112→1.3523/s，反而下降約 4.2%。
這正是不能只採用 archive 吞吐量判斷效能的原因。
原記錄 `results/L86_half_shift_observation_ide_20261007/report.json` 保留未修改。
之後才加入上述精確 energy 選擇，另行比較，沒有用比較差的 plain 結果聲稱達成目標。

native energy 的另外測試涵蓋 L6、14、22、86、94 的 960 個指定內容／row-order／compression 查詢，
每一個都用 Python 整數 PACF 確認 energy 及真正修改數。
L6 全部 input pairs 又檢查一次 energy-aware 結果：距離同為最小且 energy 不大於 plain。

第四輪使用 energy-aware 選擇，仍為 IDE 16,384 lanes、2000 步、每 25 步觀測，
同一 seed 的五次交錯量測。原方法 eligible C seeds=42，新方法=93；
formation/export 合計的中位 eligible seed throughput 1.37806→2.57147/s，增加 86.60%。
formation 本身 29.5238→34.9791 秒；原 seed bank 78 筆、新 seed bank 達上限 128 筆。
原始量測保留於 `results/L86_half_shift_energy_observation_ide_20261007/report.json`。
這是增加有效必要條件種子的工作量比較；五次為相同軌跡的計時重複，並非五個獨立 seed。
兩側完整新解都為零、最佳 score 同為 45，仍不能聲稱新解 discovery 快 20%。

最終版真實 MPS 還確認 compression=False 與 True：
兩種設定的 raw Adam／停滯狀態皆與回退版相同，新增 payload 均通過整數驗證。
記錄 `results/L86_minimum8_energy_mps_verification_20261007/report.json`。
full-only 模式採 E_full 選擇，同時讓原 payload 的 target energy 尊重此設定。

完整回歸：1300 passed、14 skipped；沿用已知兩個環境 benchmark modules 排除項。
另有兩個 boundary-expansion 的獨立測試通過，與上述回歸分開執行。
