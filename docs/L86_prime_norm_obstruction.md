# 單一integer cyclic norm的p平方Hensel必要條件

這是本專案自行推導。它適用於單一integer row的自相關，
包含binary support及signed trit rows；不適用於兩個未知row的norm總和。

令d=43、p為奇prime且p≡-1 mod43。primitive43 root α在F_(p²)，
α^p=α⁻¹。對integer cyclic row U的autocorrelation N=UU*，
N(α)=U(α)U(α⁻¹)是finite-field norm。
若N(α)=0 modp，則U(α)=0 modp，且conjugate亦為0。
將primitive根／self-reciprocal quadratic factor唯一Hensel lift到modp²，
兩個U分量均在p理想中，因此N值必為0 modp²。
故「zero modp但nonzero modp²」是嚴格不可能證書。

## 可直接用整數求值的real roots

令t=α+α⁻¹，L₀=2、L₁=t、L_u=tL_(u-1)-L_(u-2)。
real cyclotomic polynomial為 Ψ43(t)=1+Σ_{u=1}^{21}L_u(t)。
對稱N的值就是 N₀+Σ_{u=1}^{21}N_u L_u(t)。

建F_(p²)=F_p[√D]，D經Euler criterion核對為nonsquare。
primitive α由norm-one subgroup構造，核對order43及norm1；
α^j的traces给全部21個不同real roots，沒有只挑一個root。
simple root t的lift為
t+p·[-(Ψ43(t) modp²)/p·(Ψ43'(t))⁻¹ modp]。
所有lift核對Ψ43=0 modp²及t²-4為nonsquare modp，證明quadratic residue factor確實irreducible。

本版遍歷p≤50,000的所有p≡-1 mod43 primes，C整數dot products；
負係數可直接處理，不需要PSD或float。
每個證書包含profile、prime、lifted root、nonzero p² residue；
Python獨立核對primality、lifted polynomial、irreducibility及profile求值。
回傳None只表示這批有限primes沒有排除，不表示profile可實現。
公開API亦保留native=False參考及大integer fallback。

## 絕不能用於兩個未知norm總和

counterexample：trits(1,1,0)與(1,1,-1)的自相關分別(2,1,1)、(3,-1,-1)。
兩者都有效，其sum為(5,0,0)，在d=3,p=5會被「單一norm」條件排除。
這恰說明不能直接接到本專案同時求兩個unknown compressed signs的sum-PACF solver。
只有一列已固定、另一列的完整autocorrelation已知時才能使用。

目前已實作 `solver/prime_norm_obstruction.py`、`csrc/prime_norm.c`，
並接入 `complete_binary_intersection_profile(...,prime_norm=True)`，預設False可回退。
small binary／ternary全部正例保留，native／reference證書比對及L43 integer trit正例通過。
實際L86篩除比例及配對加速仍需測量，不能以此宣稱找到新解或達成20%。

## 實際閉合ledger的必要profile核對

`L86_prime_norm_profile_audit_20261008` 已重建原48,648個ordered fixed-triple fibers，
逐筆確認input、flip masks、順序及完整status，actual norm visits與parent report一致。
全部2,585個不同、PSD可行的必要profiles中，p≤50,000排除823個（31.8375%）。
全部823個證書再次独立核對；五個有真binary periodic models的profiles全部保留。
按parent實際profiles的出現次數加權，必要norm visits可排除40,393,125／126,970,443，
即31.8130%。這是profile／必要枚舉量，不是wall-time或PQCP新解發現率。

其中w=20的實際L86 profile通過所有integer range、sum及有理PSD，
却在p=859、lifted root593145，求值548901（zero mod859、nonzero mod859²）。
原完整CRT enumeration無models；開啟新條件後visited=0，完整結果相同。
此例及全部小binary profiles的完整模型集合已加入測試。
