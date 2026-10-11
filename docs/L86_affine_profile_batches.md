# 相同內容族的稀疏 affine 計算合併

固定三個實際半列，未知 binary support 為 U，長度 d=43、重量 w。
令 λ 為完整 DC／ALT 內容條件決定的整數，N_fixed 為三個已知列的 intersection profiles。
定義 B₀=w，B_u=λ−ΣN_fixed(u)（u≠0）。
even κ=2τ 的必要範數是

`N_U = B + η(δ_τ + δ_−τ)`。

odd κ 的 even 方程是 N_U=B；其 centered cross trace 方程
`D = D_base + η(δ_σ + δ_−σ)`，σ=(κ−d)/2 mod d。
因此不同目標只改動稀疏係數，不必重新計算整個 spectral dot product。

## 同一個 PSD 證書

令 c_{j,u} 為原來 outward rational cosine 上界的整數 numerator，scale=2⁴⁰。
每個原始 profile 的上界 numerator 精確等於

`Q_j(N_U)=B₀·scale+2ΣB_u c_{j,u}+2η c_{j,τ}`。

先共用第一個內積，再加 sparse delta，與逐一計算得到的整數完全相同。
B 可以有負係數；這裡只使用線性恆等式，**不主張 Q_j(B) 自身是 PSD 上界**。
所有 actual N_U 先經原始 prime-length intersection range／sum 條件，
actual coefficients 非負才有 outward upper bound 的不等式。
仍保留逐一方法的第一個負 frequency，零上界不刪除。
每個 native numerator 用 Python 整數線性式獨立重算；同一 frequency 的 signed base dot 只重算一次。

實作 `sparse_profile_psd_obstructions`、`pqcp_sparse_profile_psd`；
`batched_psd=False` 回退。小長度全部 bounded symmetric signed bases與其合法 profiles，
證書逐欄等同原始 Python 方法；130-target跨chunk及true norm零邊界也通過。

## 有限域檢查先於 PSD

在每個 F_(257²) primitive-root 分量，令 t=α+α⁻¹，L_u(t)=αᵘ+α⁻ᵘ。
共用 n_base=B₀+ΣB_u L_u(t)、h_base=D₀+ΣD_u L_u(t)、w=Norm(W)。
even κ 只需 `n=n_base+ηL_(κ/2)(t), h=h_base`；
odd κ 只需 `n=n_base, h=h_base+ηL_σ(t)`。
逐目標檢查 Δ=h²−4nw 的原始共軛二次根條件。
這些 mod-p 線性式與逐一 actual-profile 求值完全相同，
每個證書仍攜帶 actual integer N_U、literal W及target，可獨立核對。

只有證明必要条件才能刪除，不依賴先做 PSD；
因此先排除 finite-field 不相容目標，再對剩下的 profiles 做 PSD、Gram與完整 norm／trace求解。
zero partner 且 zero trace 仍保留原來完整 fallback。
不把「完整目標全部被排除」描述成「unconstrained periodic norm 沒有 preimages」。

core 選項 `finite_trace=True, affine_finite=True`；`affine_finite=False` 回退至 PSD先行逐profile版本。
最初將even／odd全部提前的版本，完整三次配對反而慢39.84%，
因odd targets共用一個norm時製造了大量不必要的trace certificates。
因此當前core的affine_finite只提前even targets，odd groups仍先做單一PSD判定。
這是必要條件順序及計算共享的調整，不以經驗結果作UNSAT條件。
全部 L6 literal 固定三列與原始全 binary PQCP 空間核對，包含零列、所有 free roles、
norm及GF(2)trace求解。d=3/5/11/43 的 affine結果逐target與原始finite filter比對，
全部 native exclusions 附獨立 Python 證書核對。

## 計時的限定

`results/L86_sparse_psd_paired_20261008/report.json`：
相同48,648個fixed triples、三次交錯warm replay，full-word outputs及原始targets一致，
PSD證書也完全一致；baseline與candidate都已開啟原始finite／Gram條件。
中位數80.189528667→69.826686417秒，local時間減少約12.92%。
每個版本每輪6,325個finite證書，共37,950個，全部獨立核對。
兩端完整新解均零；不是20%新解發現率證據。
後續 affine-finite 重排的獨立配對以其 report 的 complete 欄位和最終數據為準。

全部targets提前的舊快照完整記錄於
`results/L86_affine_finite_paired_20261008/report.json`。
median12.892559458→18.02915575秒，完整輸出集合一致，
3,083,799個finite certificates全部獨立核對；兩端新解零。
該較慢版本沒有設為IDE預設。
後續even-only配對保存到 `results/L86_affine_finite_even_paired_20261008/`。
目前實驗IDE預設已開啟證書完全相同且local改善12.92%的batched PSD；
core仍可關閉，affine先行順序則尚未預設開啟。

even-only版本的三次配對也已完成：
median10.405686541→13.749403042秒，local反而慢32.13%；
1,558,128個finite certificates全部獨立核對，完整full-word集合與targets一致，novel classes仍零。
雖減少odd目標證書，early finite仍產生大量本來一個PSD證書即可排除的分支。
因此當前IDE預設保留PSD先行與batched PSD，`affine_finite=False`；
新增API作為完整、可回退的研究版本，不宣稱其已改善速度。
更大region使用哪個版本，由其凍結source及report選項決定，不能用後來的預設覆寫既有結果。
