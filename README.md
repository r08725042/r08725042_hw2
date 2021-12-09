# ADL HW2 README

(使用 python 3.6)

1. 執行 download.sh，下載model weight

2. 使用Model
    - 如果"不重新train model"，直接進行預測
        - 執行 'run.sh'，並輸入test data的位置及檔案輸出位置，此部分會包含 test data的前處理，執行完畢後將output出預測結果
    
    
    - 如果"要重新train model"
        1. 將資料放入 data/ 資料夾中(路徑寫死)
        2. 執行bert_train.py


3. 檔案介紹
    1. preprocessing_data.py : 裡面包含preprocessing_data會用到的function，此一檔案會被import至bert_train.py及 bert_predict.py

    2. bert_train.py : 用於訓練model的檔案

    3. bert_predict.py : 用於進行預測的檔案

    4. hw2_plot1.py : 用於繪製 Answer length distribution 的檔案

    5. hw2_plot2.py : 用於繪製 Answerable Threshold 的檔案

    6. bert_weight.pt : model的weight，由download.sh下載

