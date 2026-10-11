# F_(257²)的共軛二次根必要條件

這是本專案自行推導，將 [norm／trace補全](L86_norm_trace_pruning.md) 的條件
投影到另一個characteristic，不修改binary字母表或原始接受條件。
43為prime，257≡-1 mod43，因此primitive43 roots在F_(257²)，star就是257次Frobenius。
real cyclotomic Ψ43(t)=1+Σ_{u=1}^{21}L_u(t) 在F257有全部21個simple roots，
其中L₀=2、L₁=t、L_u=tL_(u-1)-L_(u-2)。全部roots都保留，無sampling。

令未知binary support為U、literal companion為W，
given norm n=UU*及given anchored cross trace h=WU*+W*U。
在每個quadratic field令z=UW*，則

\[
z+z^*=h,\qquad zz^*=n\,\operatorname{Norm}(W),\qquad
\Delta=h^2-4n\operatorname{Norm}(W).
\]

若Norm(W)非zero、Δ是**非zero square**，二次方程的兩個不同roots都在F257。
但Frobenius固定這些roots，無法將它們互相交換，不能同時滿足trace／norm。
因此該component嚴格不可能。Δ=0保留唯一root；nonsquare保留共軛root pair。
若Norm(W)=0，因residue ring是field而W*=Frobenius，必有W=0。
這時h≠0排除，h=0保留所有norm possibilities，不加入額外假設。

## 從integer profiles求值

所有對稱profile F的值是 F₀+Σ_{u=1}^{21}F_u L_u(t) mod257。
N_U來自even位移，N_W是literal W的完整integer intersection profile。
cross base為D_base=λ-C_other；odd κ的target增加ηL_τ(t)，τ=(κ-43)/2 mod43，
even κ沒有這個odd delta。
C以integer dot products及完整nonzero-square lookup逐一判斷所有targets。
每個被刪target附prime、real root、discriminant及原始profiles；
Python獨立核對Ψ43(root)=0、quadratic factor irreducible、完整profile值及Legendre criterion。
計算不使用float、epsilon、近似PSD或GPU接受判定。

這是given SINGLE unknown norm及其known companion trace的條件；
不是任意兩個unknown norm相加的判定。
它與 [p²單一norm障礙](L86_prime_norm_obstruction.md) 不同：
後者利用zero norm的偶數valuation，這裡也可排除nonzero norm／trace的不相容。

## 覆蓋、測試與回退

小長度全部binary U/W、正負eta、zero partner均保留真integer cross targets；
L11全部literal U及L43抽樣integer正例通過。
core的全部L6固定三列、全literal PQCP輸出集合也與original oracle一致，
每個native exclusion另外獨立重算證書。
接受者仍要完整integer intersections／odd equations及原始86點verifier。

`finite_trace=False`可回退，core預設False；
`experiments/benchmark_two_half_gram.py --comparison finite`比較相同原始fibers的full-word集合，
全部排除證書在計時之外核對。配對數據及IDE預設決定以狀態文件的最終記錄為準。
任何零對零的新解結果或norm visits减少，都不能計算完整新解發現速度提升。
實作 `solver/finite_field_trace_pruning.py`、`csrc/finite_field_trace.c`。

## 完成的長度86配對

`results/L86_two_half_finite_trace_paired_20261008/report.json`完成三次交錯warm replay。
每輪48,648個相同fixed triples，原始full-word輸出集合與所有必要targets完全一致。
每輪6,325個targets全部由finite條件排除，norm visits 126,970,443→0。
三輪共18,975個排除證書在計時之外全部Python獨立核對。
median baseline94.230282209、candidate81.056975291秒，local時間減少13.9799%。
這個百分比僅來自該次配對，不能和其他run跨次比較。
兩端verified novel classes都零，尚無20%完整新解加速證據。

研究實驗入口 `complete_two_l86_halves.py` 已預設開啟原始finite及joint Gram條件；
core API仍預設False。這不表示`main.py`的原始MPS搜尋已改成這種完整半列補全。
後續 [affine合併與順序重排](L86_affine_profile_batches.md) 再利用其線性求值結構，
保留同一數學條件及每個actual-profile證書。
