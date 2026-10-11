# 聯合自相關／交叉相關的嚴格頻譜Gram剪枝

這是本專案從整數cyclic correlations自行推導的必要條件。
U是未知physical binary negative support，W是同整列的已知centered半列。
完整even equations給N_U，known W給N_W，odd equations給對稱cross counts D。
在integer cyclic ring中 D=UW*+U*W，所以

\[
\widehat D(j)=2\operatorname{Re}(\widehat U(j)\overline{\widehat W(j)}),\qquad
\widehat N_U(j)=|\widehat U(j)|^2,\quad\widehat N_W(j)=|\widehat W(j)|^2.
\]

由實二維向量的Gram matrix必為PSD，得到

\[
\widehat D(j)^2\le4\widehat N_U(j)\widehat N_W(j).
\]

它比只要求unknown PSD非負更強：一個可實現的N_U及可實現的N_W，
仍可能不允許所需D。測試有整數range、sum及diagonal parity均符合、
兩個norm profiles都PSD的cross例子，由此新条件嚴格排除。

## 不使用float的向外區間

[Machin π的有理上下界](L86_certified_profile_psd.md) 已給cosine上界。
另在x=2π_upper phase/43使用cosine的odd Taylor截斷至n=17，
因x<π_lower<π、cos遞減及交錯尾項單調，可得true cosine的下界。
下界以floor、上界以ceil乘S=2⁴⁰，所有端點都是integer／S。
positive coefficients與negative coefficients分別選正確的端點，故signed D也有嚴格區間。

令U_j、W_j為兩個norm Fourier的上界分子，D Fourier的區間為[L_j,H_j]/S。
若unknown PSD上界負，直接不可能；known literal W的上界不能負，否則程式報錯。
令m_j=0若區間包含0，否則m_j=min(|L_j|,|H_j|)。則

\[
4\widehat N_U\widehat N_W-\widehat D^2\le(4U_jW_j-m_j^2)/S^2.
\]

右側分子嚴格負才排除；零或正只表示沒有排除。
C用signed64位做dot products、signed128位做平方／乘積，輸入界已證明不溢位。
每個native證書可由 `verify_joint_gram_certificate` 用Python任意精度整數獨立重算。

## 同時測所有targets

在centered gauge，D_base(u)=λ-C_other(u)。
odd κ的target在±τ加η，其中τ=(κ-43)/2 mod43；even κ不增加odd delta。
因此每個frequency只需先算D_base區間一次，再用2ηcos(2πjτ/43)調整。
全21個odd targets被一起判斷，實際bit phases及zero-partner norm／trace回退仍完整保留。
只有所有targets均被證書排除時，才跳過該profile的完整norm enumeration。
這時沒有宣稱無constrained-target以外的periodic norm preimages。

## 配對、輸出與範圍

`L86_two_half_gram_paired_20261008`：同八個中心、48,648個eligible fixed-triple fibers，
三次暖機交替順序，兩側共用相同native individual PSD及profile templates。
每個fiber的全部完整words、必要profile／原始target集合完全一致。
候選側每回放6,325個targets，排除6,324個；norm visits 126,970,443→49,923。
三次共18,972個joint certificates在計時之外逐一由Python獨立重算，全部通過。
中位16.111819708→13.570192542秒，局部减少15.7749%。
兩側完整新等價類均零，所以這不是完整新解發現速度，更未達成20%目標。
分開執行的其他wall times不作配對證據；所有source／input／初始inventory另存。

small-length所有literal U/W的真cross都保留；所有L6固定三列的全部解集合也保留。
core的joint_gram預設False可回退，IDE兩半列實驗預設True。
程式 `solver/joint_gram_pruning.py`、`csrc/joint_gram.c`。

individual PSD另將相同integer dot products交給C，所有負分子仍由Python獨立重算。
`profile_psd_obstruction(...,native=False)`保留原參考；任意大integer inputs自動走Python。
native／reference在全部小nonnegative symmetric profiles的證書完全一致。
