# 固定一個半列、改動另外三個半列

兩半列搜尋固定兩個 literal rows；即使完整完成，也不涵蓋要同時改動三列的解。
現在以四個 physical slots `(A_even,A_odd,B_even,B_odd)` 直接推導更大的有限區域。

選一個 free slot f，其全部43bits與重量都自由。
從另外三列選兩個 varied slots x<y，剩下的一列保持 literal 不變。
兩個 varied rows分別 XOR非零 masks X、Y，並要求
`1≤|X|, 1≤|Y|, |X|+|Y|≤r`。
所有位置均允許；不限制為連續區塊、保重量交換或特定κ／η。
每個完整 known triple 都交給既有 exact completion，保留所有必要target contents與所有輸出phases。

4·C(3,2)=12種 roles，各中心的原始 triples 數是

`12 Σ_(a≥1,b≥1,a+b≤r) C(43,a)C(43,b)`。

r=2時22,188；r=3時954,084。
XOR對mask是單射；因兩mask都非零，literal triple中恰兩個 known rows 改動，
因此反推出唯一的varied pair，角色間沒有同一fixed triple重複。
這不代表不同free roles輸出的**完整解**不重複；writer仍按完整PQCP等價類去重。
全部12roles及任意位置總Hamming距離在row shifts／parity交換、reflection、common decimation下有完整映射，
不需假設同PAF的不同literal rows可互換。

`solver/three_half_row_completion.py`是generator，
`experiments/complete_three_l86_halves.py`可直接在IDE按執行鍵：
預設從最新root L86 long-test log選八個已核對非解中心，total radius=3。
中心排名是啟發式選擇，並非其他centers被數學排除；完整範圍以frozen input／source及report為準。
可回退到原本 `complete_two_l86_halves.py`，仍預設一個 bounded varied half。

全部L6解先做literal原始驗證，再對三個非解centers及r=2/3/4逐一核對完整區域。
generator計數、eligible literal keys唯一性及所有完整輸出集合均與oracle一致。

## 已完成與正在進行的區域

`results/L86_three_halves_total2_top8_20261008/report.json` complete=True。
八個最新長測中心，177,504原始triples、32,600 eligible fibers。
385,080個profiles完整判定，383,850附PSD排除證書，2,670個targets被原始finite必要條件排除。
沒有norm visits或完整新解；elapsed=87.640322333秒。
是明確有限区域排除，不能證明L86一般不存在。

接著r=3的完整區域已啟動，來源／input凍結到
`results/L86_three_halves_total3_top8_20261008/`。
此run採用已核對的affine finite-field先行條件；是否完成以report.json為準。
沒有時間抽樣截止、沒有preimage或完整witness上限。
新解為零時不能用此類排除計數或runtime聲稱新解發現速度提升20%。

r=3後續也已完整完成八個centers，原始triples=7,632,672，
elapsed=645.323959167秒，沒有完整新解。
其frozen版本是even及odd全部finite提前的舊研究版本，
後來配對顯示該順序較慢，因此新IDE入口預設回到PSD先行。
不能用後來程式碼的even-only選項重新解釋這次凍結版本。

## 四個半列都允許改動

新增 `four_half_row_fibers`：選free slot f，另外三列皆XOR非零 masks，
總翻轉數≤r；free列全部43bits任意，沒有一個literal row固定。
全部四個free roles保留，原始triples數為

`4 Σ_(a,b,c≥1,a+b+c≤r) C(43,a)C(43,b)C(43,c)`。

r=3是4·43³=318,028 triples／center，八個centers總數2,544,224。
三個非零known-row masks仍有XOR單射性，因此literal fixed triple keys唯一。
free列可以與原始center相同，故完整輸出仍可能與三半列區域重疊；由完整等價key去重。
所有L6的三個centers、r=3/4的declared-region完整解集合已與literal oracle核對一致。

IDE入口 `experiments/complete_four_l86_halves.py` 直接按執行鍵即可，
預設最新root長測log、八個audited非解centers；radius3完整完成後，當前IDE預設改為total radius4，
PSD先行及已核對的batched PSD／finite／Gram。
實驗已啟動於 `results/L86_four_halves_total3_top8_20261008/`；
是否完成以report.json為準。此擴張仍非全域搜尋，不聲稱新解加速率。

四半列r=3後續已完整完成：2,544,224個triples、667,040 eligible fibers，
7,340,720個profiles，其中7,325,344個有PSD證書、29,644個targets被finite條件排除。
norm visits及完整新解均零；elapsed=197.773829125秒。

接著r=4完整子區域以四個獨立CPU程序啟動於
`results/L86_four_halves_total4_parallel_20261008/`。
每center20,353,792個raw triples，全部八個centers的預定總數162,830,336。
它使用fresh模式，沒有把舊r=3的closed結果混進本次完成計數；
r=4本身包含r=3，仍須每個child完整結束才計入complete。
來源凍結、gzip完整proof ledger、逐center完整計數及獨立全長verifier保留。
不以時間觀察窗或短測沒有新解而終止這個更大區域。

後續 r=4 的八個 child 已全部完整結束：162,830,336 個 raw triples，
23,024,744 個 eligible fibers，完整新解零。
原 coordinator 在彙整時誤比較 JSON list 與記憶體 tuple，並非 child 搜尋失敗。
`reconcile_parallel_completion.py` 已重新核對全部 literal inputs、source hashes、
宣告計數及完整解庫，原失敗 manager report 另存，未重跑已閉合區域。
目前 report complete=True、credited indices=0..7。
原 coordinator elapsed 不是完整總時間，因此 reconciliation 將 elapsed 設為 null，
不能用這個混合證據宣稱配對加速率。
