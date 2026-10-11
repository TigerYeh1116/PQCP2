# radius-four完整區域的程序重排與證據

原始 `L86_latest_two_halves_gram_radius4_top8_20261008` 的前兩個中心已完整結束，
每個1,640,376個role／flip指派、658,654 eligible fibers，全部無完整新解。
其report只計入closed prefix；第三個尚未完成中心不計入。

新manager固定保存該closed prefix、初始inventory、source manifest及完整八個input centers，
從另一份frozen source以四個獨立CPU程序完整完成剩餘六個中心。
compiled binaries／pycache写入runtime copy，權威source snapshot保持不變；
264個manifest entries的SHA256已全部核對。
原始PID5935確認屬本對話實驗後終止，console另保存，沒有刪除資料或重跑已closed的中心。
原始partial parent仍不是complete；新manager才合併互不重複的center indices。

首次manager因relative runtime path在changed cwd下重複拼接而未啟動任何搜尋，
錯誤log保留於`L86_two_half_parallel_remaining_20261008`。
改為absolute paths後的v2才是有效執行；不能將v1的未完成report算入覆蓋。

## 最終有限區域結果

`results/L86_two_half_parallel_remaining_v2_20261008/report.json` complete=True，
credited center indices為0..7；前兩個來自pinned parent，其餘六個child各自complete。
總共13,123,008個role／flip指派、5,530,424 eligible literal fixed triples，完整新解零。
六個新children有184,281個finite-target exclusions；所有final full models都需原始86點verifier。
manager elapsed=1507.813987542秒，不含原始前兩中心計算時間。
前兩中心Gram-only，後六中心finite+Gram；完整必要條件保留同一full解集合。
不能把此不同版本／混合歷史的總時間當作單一方法配對加速率。

對這八個input centers，所有兩半列region：一列距離≤4、另一列全部43bits任意、其他兩列固定，
全部12ordered roles已被完整判定。
因此若完整解與其中某center只差兩個半列，那兩列距離都必須至少5。
這是該center附近的嚴格距離結論，不是一般不存在定理。
它支持繼續 [三半列更大region](L86_three_half_region.md)，而不是只改同一固定三列的空fibers。

## 四半列 total radius 4 的彙整核對

`L86_four_halves_total4_parallel_20261008` 的八個 child 均完整完成；
162,830,336 raw triples、23,024,744 eligible jobs、完整新類零。
coordinator 在最後 list／tuple 比較出錯，已保留原 report，
由 `experiments/reconcile_parallel_completion.py` 重新核對輸入、來源 SHA、
每個 child 完整計數、完整解庫及排除範圍後寫入 reconciliation。
complete=True 是八份完整 child 的聯集結論；沒有重跑或捏造完成時間。
elapsed=null，原 coordinator 時间不能當完整區域配對效能證據。
