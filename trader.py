import csv
import random
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score;

def load_data(dataPath):
    ''' ### Data Read-in
        dataPath:   [str] Location of the data. Data should be in CSV Format.

        ### 資料讀入
        dataPath:   [str] 資料來源位址，資料需要是 CSV 格式。
    '''
    with open(dataPath, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f);
        data = [[float(cell) for cell in row] for row in reader];
        return data;


class Trader():
    ''' ### Auto Trader
        #### Properties - Testing Environment
        _current:   [int] Current number of stock hold during testing phase
        _testData:  [list] The acummulated test data record, to be feeded day by day in chronical order
        _actions:   [list:int] The stored predicted actions after the test prediction runned based of the trained model
        _cumGain:   [float] The cummulative gain during testing based on the predicted actions
        _currentPrice:  [float] The initial price of any hold/sold stock
        _accLoss:   [tuple/int] A rational checking constraint accumulator
        _observeDays:   [int] The number of days including feed-in today to be the observation period

        ### 自動交易
        #### 屬性 - 測試環境變數
        _current:   [int] 測試時當下的股票持有量
        _testData:  [list] 在導日順時每日資料後的累積測試資料
        _actions:   [list:int] 基於訓練模型，在進行預測後的累積動作
        _cumGain:   [float] 在測試時基於預測動作而總累積的收入
        _currentPrice:  [float] 任何時間下的當下持有股票之起始值
        _accLoss:   [tuple/int] 理性行動檢測器
        _observeDays:   [int] 包括新導入日數的觀察資料日數
    '''
    def __init__(self):
        self._current = 0;
        self._testData = [];
        self._actions = [0];
        self._cumGain = 0;
        self._currentPrice = 0;
        self._observeDays = 14;
        self._accLoss = (0,0);
        pass;

    def reset(self):
        self._current = 0;
        self._testData = [];
        self._actions = [0];
        self._cumGain = 0;
        self._currentPrice = 0;
        self._accLoss = (0,0);
        pass;

    def resultToActions(self, rawData, situation, printActions=False, name="", initial=0, initialAction=0, initialCumGain=0, initialCurrentPrice = -1, initialAccLoss = 0, additional = True):
        ''' ### Training Results to Actions
            The training is a prediction of future stock trend, and need to convert to action.
            #### Parameters
            rawData:    [list] Data with original column format
            situation:  [tuple] The 3 columns of situation estimations
            printActions:   [bool] Whether to print the accumulated action list
            name:   [str] The file name for the action list
            initial:    [int] The initial stock count
            initialAction:  [int] The previous action
            initialCumGain: [int] The current cummulative gain
            initialCurrentPrice:    [int] Initial price of any stock on hold/sold
            additional: [bool] Whether to add manual rational rules

            ### 訓練結果轉換至動作
            訓練結果是對股票往後日子的境況預測，需要把結果轉換為動作。
            #### Parameters
            rawData:    [list] 原始資料，使用資料來源的欄位格式
            situation:  [tuple] 處境判斷的3個欄位
            printActions:   [bool] 是否要列印記載的動作列表
            name:   [str] 需要列印動作列表的檔案名稱
            initial:    [int] 股票數量的起始值
            initialAction:  [int] 上一個動作
            initialCumGain: [int] 目前的累積收獲
            initialCurrentPrice:    [int] 持有或放售投票的原值
            additional: [bool] 是否加入手動理性調整條件
        '''
        # 把導入參數重新整理
        current = initial; actionCol=[initialAction]; currentPrice = initialCurrentPrice; cumGain=initialCumGain;
        nextRiseC, nextNextRiseC, toAmplifyC, accuracy = situation;  accLoss, accLoss2 = initialAccLoss;

        # 在處境的每行準備處理
        for idx, nextRise in enumerate(nextRiseC):
            # nextRise: 明天會上升?
            nextNextRise, toAmplify = nextNextRiseC[idx], toAmplifyC[idx];
            
            # 處理上個日的動作
            toAction = actionCol[idx];
            if (current == 0):
                if (toAction == 1): 
                    currentPrice = rawData[idx][0];
                    current = 1;
                elif (toAction == -1):
                    currentPrice = rawData[idx][0];
                    current = -1;
            elif (current == 1):
                if (toAction == -1):
                    cumGain += rawData[idx][0] - currentPrice;
                    currentPrice = -1;
                    current = 0;
                elif (toAction == 1):
                    print("出現錯誤，動作 #", i)
            elif (current == -1):
                if (toAction == 1):
                    cumGain += currentPrice - rawData[idx][0];
                    currentPrice = -1;
                    current = 0;
                elif (toAction == -1):
                    print("出現錯誤，動作 #", i)

            # 從目前境況判斷動作
            if (current == 0):
                if nextRise:
                    action = 1; 
                elif not nextRise:
                    action = -1;
            elif (current == 1):
                if nextRise and nextNextRise:
                    action = 0;
                elif nextRise and (not nextNextRise) and toAmplify:
                    action = -1;
                elif nextRise and (not nextNextRise) and (not toAmplify):
                    action = 0;
                elif not nextRise:
                    action = -1;
            elif (current == -1):
                if (not nextRise) and (not nextNextRise):
                    action = 0;
                elif (not nextRise) and nextNextRise and toAmplify:
                    action = 1;
                elif (not nextRise) and nextNextRise and (not toAmplify):
                    action = 0;
                elif nextRise:
                    action = 1;

            # 加入人為因素觀察
            if (additional):
                possiblyNextPrice = rawData[idx][3];
                accLossThreashold = 3; accLossThreashold2 = 10;
                if (action == -1 and current == 1 and (possiblyNextPrice - currentPrice) < 0):
                    if (accLoss2 < accLossThreashold2):
                        accLoss2 += 1;
                    else:
                        action = 0;
                        accLoss2=0;
                elif (action == 1 and current == -1 and (currentPrice - possiblyNextPrice) < 0):
                    if (accLoss2 < accLossThreashold2):
                        accLoss2 += 1;
                    else:
                        action = 0;
                        accLoss2 = 0;
                elif (action == 0 and current == 1 and (possiblyNextPrice - currentPrice) < 0):
                    if (accLoss > accLossThreashold):
                        action = -1;    
                        accLoss=0;
                    else:
                        accLoss += 1;
                elif (action == 0 and current == -1 and (currentPrice - possiblyNextPrice) < 0):
                    if (accLoss > accLossThreashold):
                        action = 1;
                        accLoss=0;
                    else:
                        accLoss += 1;
                
            # 在動作欄加入這日的動作
            actionCol.append(action);
        
        # 如需要列印動作，則儲存至 CSV 檔案中
        if (printActions):
            with open(name+'.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
                toWriter = csv.writer(csvfile);
                for idx, row in enumerate(actionCol[1:]):
                    toWriter.writerow([*(rawData[idx][:4]), row]);
        
        # 回傳每日的動作，目前手持股票量，目前累積收穫，目前手持股票原值
        return actionCol[1:], current, cumGain, currentPrice, (accLoss, accLoss2);

    def testActions(self, data, actions):
        ''' ### Test Actions
            Given the original data and the action list, try and see how the auto trader performs.
            data:    [list] Data with original column format
            actions:    [list:int] Action List as defined by the homework requirement

            ### 測試動作
            在原始資料和建議的動作下，嘗試並觀察自動交易器會有怎樣的結果
            data:   [list] 原始資料，使用資料來源的欄位格式
            actions:    [list:int] 動作列表，定義依照作業要求
        '''
        # Buy and Hold Strategy 的收獲
        compareTo = data[-1][3] - data[1][0];
        print("Buy and Hold Strategy: ", compareTo);

        # 測試值初始化
        currentPrice = -1;
        cumGain = 0;
        current = 0;

        # 在第二天開始計算每個動作的結果
        for i in range(1, len(data)):
            toAction = actions[i-1];
            if (current == 0):
                if (toAction == 1): 
                    currentPrice = data[i][0];
                    current = 1;
                elif (toAction == -1):
                    currentPrice = data[i][0];
                    current = -1;
            elif (current == 1):
                if (toAction == -1):
                    cumGain += data[i][0] - currentPrice;
                    currentPrice = -1;
                    current = 0;
                elif (toAction == 1):
                    print("出現錯誤，動作 #", i)
            elif (current == -1):
                if (toAction == 1):
                    cumGain += currentPrice - data[i][0];
                    currentPrice = -1;
                    current = 0;
                elif (toAction == -1):
                    print("出現錯誤，動作 #", i)
        
        # 結尾時，把所有股票歸零
        if (current == 1):
            cumGain += data[i][3] - currentPrice;
        elif (current == -1):
            cumGain += currentPrice - data[i][3];
        
        # 列印這種累積結果，比較 Buy and Hold Strtegy 的分別
        print("訓練結果: ", cumGain);
        more = (cumGain-compareTo);
        print("比較結果: " + ("+" if more >= 0 else "-"), abs(more));
        return more;



    def fitAndPredict(self, data):
        ''' ### Fit and Predict Data
            Using regression to fit the observed data, and to give predicted scenarios.
            data:    [list] Data with original column format

            ### 模型建立及預測資料
            使用迴歸方法形容觀察數據，並預測將會出現的情境
            data:   [list] 原始資料，使用資料來源的欄位格式
        '''
        # 資料標準化
        timeFrame = self._observeDays; 
        x = [[t/timeFrame, (t/timeFrame)**2, (t/timeFrame)**3, (t/timeFrame)**4, (t/timeFrame)**5] for t in range(0, timeFrame)];
        y = [row for row in np.array(data)[-timeFrame:,0]];
        normY = y[-1];
        y = [(row-normY)/normY for row in y];

        # SK-Learn 迴歸模型建立
        regr = linear_model.LinearRegression();
        regr.fit(x, y);
        predY = regr.predict(x);
        toY = regr.predict( [[t, t**2, t**3, t**4, t**5] for t in [(timeFrame-1/timeFrame), 1, (timeFrame+1/timeFrame)]]);

        # [內部測試] 趨勢的準確度
        oriTrend = np.array([row for row in np.array(data)[-timeFrame+1:,0]]) - np.array([row for row in np.array(data)[-timeFrame:-1,0]]);
        predTrend = np.array([row for row in predY[1:]]) - np.array([row for row in predY[0:-1]]);
        multi = oriTrend * predTrend;
        accuracy = len([x for x in multi if x < 0])/len(multi);

        # 回傳三個將會預測的情境和準確度
        return ([1 if  toY[1] - toY[0] >= 0 else 0], [1 if  toY[2] - toY[1] >= 0 else 0], [1 if  abs(toY[2] - toY[1]) >= abs(toY[1] - toY[0]) else 0], accuracy);

        
        
    def predict_action(self, datum, additional = True):
        ''' ### Trader Action Prediction
            datum:  [list] A single day newly observed, elements: open, high, low, close prices
            additional: [bool] Whether to add manual rational rules
            ### 交易模型動作預測
            datum:  [list] 新觀察的單日報價資料，欄位：開價,高位,低位,收市價
            additional: [bool] 是否加入手動理性調整條件
        '''
        # 把新增資料記錄下來
        self._testData.append(datum);

        # 把新增資料記錄下來
        if (len(self._testData) >= self._observeDays ):
            situation = self.fitAndPredict(self._testData);

            # 把所有的處境預測推回動作
            toActions, self._current, self._cumGain, self._currentPrice, self._accLoss = self.resultToActions(self._testData, situation, printActions=False, initial = self._current, initialAccLoss = self._accLoss, initialAction=self._actions[-1], initialCumGain=self._cumGain, initialCurrentPrice=self._currentPrice, additional = additional );
            self._actions.append(toActions[0]);
            # 回傳當下動作
            return self._actions[-1];
        else:
            self._actions.append(0);
            return 0;



# You can write code above the if-main block.

if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.

    print("匯入資料 - 訓練資料")
    training_data = load_data(args.training)
    trader = Trader();

    additional = True;
    print("匯入資料 - 測試資料")
    testing_data = load_data(args.testing);
    trader.reset();
    print("判定動作中 (請稍後) ...")
    with open(args.output, 'w') as output_file:
        for idx,row in enumerate(testing_data):
            # 除了最後一天，都會進行動作預測
            if (idx<len(testing_data)-1):
                # We will perform your action as the open price in the next day.
                action = trader.predict_action(row, additional = additional)
                output_file.write(str(action) + "\n")
                # this is your option, you can leave it empty.
                #i=0;
                #trader.re_training(i)
            # 以下是最後一天，不會列印，但儲存作內部測試
            else:
                if trader._current == 1:
                    trader._actions.append(-1);
                elif trader._current == -1:
                    trader._actions.append(1);
        # 內部測試結果
        googleResult = trader.testActions(testing_data, trader._actions[1:]);
    
    print("程序終結，已列印 output.csv 作動作預測");