#!/usr/bin/env python
# -*- coding: utf-8 -*-

from GA import Problem
from GA import Solution
import numpy as np
import numpy.random as random
import pandas as pd;

class stackTradeProblem(Problem):


    def __init__(self,stackData_,seedMoney_,index):
        nOfVal = 16
        super().__init__(nOfVal,1,index);
        self.division = [nOfVal]*self.nOfVariables;

        ##手数料
        self.Commision = 100;

        ## 種銭
        self.seedMoney = seedMoney_;
        
        ## 遺伝子座から実際に求められる値への変更
        self.intervals = [5,10,15,20,25,30,50,75,100,200];
        self.coefficients = [1.0,1.5,2.0,2.5,3.0];
        self.expSmootParam = [0.02,0.04,0.06,0.08,0.10];
        

        ##株を買うときの最小単位
        self.unit = 100;

        ## 各決定変数が取りうる値の総数
        self.division[0] = 10; ## チャンネルブレイクアウト(買い用)
        self.division[1] = 10; ## チャンネルブレイクアウト(売り用)

        self.division[2] = 10; ## ポリんジャーバンド (買い用:パラメータ 1)
        self.division[3] = 5; ## ポリんジャーバンド (買い用:パラメータ 2)
        self.division[4] = 10; ## ポリんジャーバンド (売り用:パラメータ 1)
        self.division[5] = 5; ## ポリんジャーバンド (売り用:パラメータ 2)

        self.division[6] = 10; ## 単純平均移動   (買い用:長期)
        self.division[7] = 10; ## 単純平均移動   (買い用:短期)
        self.division[8] = 10; ## 単純平均移動   (売り用:長期)
        self.division[9] = 10; ## 単純平均移動   (売り用:短期)

        self.division[10] = 10; ## 指数平滑移動平均 (買い用:長期)
        self.division[11] = 10; ## 指数平滑移動平均 (買い用:短期)
        self.division[12] = 5; ## 指数平滑移動平均 (買い用:パーセント)
        self.division[13] = 10; ## 指数平滑移動平均 (売り用:長期)
        self.division[14] = 10; ## 指数平滑移動平均 (売り用:短期)
        self.division[15] = 5; ## 指数平滑移動平均 (売り用:パーセント)

        ## 学習データと評価用データ(DataFrame形式)
        ## 学習データは約現在1年分(大まかに設定しており，うるう年などは想定に入れていない)
        ## 学習データは約3ヶ月分(大まかに設定しており，1ヶ月は30日としてカウント)
        #self.trainData = stackData_[index*365:(index+2)*365];
        #self.testData = stackData_[(index + 2) * 365+1:(index + 3) * 365];

        self.trainData = stackData_;
        self.testData = stackData_;

        self.trainStartPoint = (index%5)*365;
        self.trainLerningInterval = 2*365;
        self.testStartPoint = (index%5) * 365 + 2*365+1;        
        self.testLerningInterval = 1 * 365;

        ## 学習用データの抽出
        self.trainOpenArray = self.trainData.get("Open").values
        self.trainCloseArray = self.trainData.get("Close").values
        self.trainHighArray = self.trainData.get("High").values
        self.trainLowArray = self.trainData.get("Low").values

        ##　評価用データの抽出
        self.testOpenArray = self.testData.get("Open").values
        self.testCloseArray = self.testData.get("Close").values
        self.testHighArray = self.testData.get("High").values
        self.testLowArray = self.testData.get("Low").values




    def repair(self,sol:Solution):
        if (sol.variables[6] == 0):
            sol.variables[6] = random.randint(0,sol.getDivision()[6]-1) + 1;
        if (sol.variables[8] == 0):
            sol.variables[8] = random.randint(0,sol.getDivision()[8]-1) + 1;
        
        if (sol.variables[7] > sol.variables[6]):
            sol.variables[7] = sol.variables[6]-1;
        
        if (sol.variables[9] > sol.variables[8]):
            sol.variables[9] = sol.variables[8]-1;
        

        if (sol.variables[10] == 0):
            sol.variables[10] = random.randint(0,sol.getDivision()[10]-1) + 1;
        if (sol.variables[13] == 0):
            sol.variables[13] = random.randint(0,sol.getDivision()[13]-1) + 1;
        
        if (sol.variables[11] > sol.variables[10]):
            sol.variables[11] = sol.variables[10]-1;
        
        if (sol.variables[14] > sol.variables[13]):
            sol.variables[14] = sol.variables[13]-1;





    ## SMA(Simple Moving Average: 単純平均移動)の計算
    ## currentData: 現在の日付
    ## interval: 単純平均移動の計算に使用する期間(5なら過去5日間の平均を求める)
    ## closeArray: 終値の推移
    def calcSMA(self,currentDate,interval,closeArray):
        buyStart = max(currentDate - interval, 0);
        return np.average(closeArray[buyStart:currentDate]);

    ## SMA(Exp Moving Average: 指数平滑移動平均)の計算
    ## currentData: 現在の日付
    ## interval: 単純平均移動の計算に使用する期間(5なら過去5日間の平均を求める)
    ## closeArray: 終値の推移
    ## SMAArray: SMAArray
    def calcEMA(self,currentDate,interval,percent,SMAArray,closeArray):
        startPoint = max(0,currentDate- interval);
        EMA = SMAArray[startPoint];
        for i in range(0,currentDate - startPoint+1):
            EMA = EMA + percent*(closeArray[startPoint+i] - EMA);
        return EMA;


    ## チャンネルブレイクアウトによる売買決定 (テクニカル手法の一つ)
    ## 1: buy, -1: sold, 0: otherwise
    ## higharray, lowarray, openArrayはそれぞれ，高値，低値，初値の推移
    ## currentDateは日付
    ## buyInterval, soldInterval チャンネルブレイクアウトを計算するためのインターバル (決定変数の一部)
    def checkByChannelBreakOut(self, currentDate, buyInterval, soldInterval, higharray, lowarray, openArray):
        buyStart = max(currentDate - buyInterval, 0);
        soldStart = max(currentDate - soldInterval, 0);

        hi = np.max(higharray[buyStart:currentDate]);
        lo = np.min(lowarray[soldStart:currentDate]);

        if (openArray[currentDate] > hi):
            return 1;
        elif (openArray[currentDate] < lo):
            return -1;
        else:
            return 0;


    ## 長期SMAと短期SMAを用いて，ゴールデンクロスかデッドクロスか判定する
    ## 1: ゴールデンクロス, -1: デッドクロス, 0: どっちでもない
    def checkGoldenOrDead(self,longSMA,shortSMA):       
        ##二つの関数の差分が全て正or負なら，二つの関数は交わっていない．
        long = np.array(longSMA);
        short = np.array(shortSMA);
        ret = long > short;

        if (np.all(ret)):
            return 0;
        elif (np.all(~ret)):
            return 0;

        ## 最後のものを見て比較をする．
        if(shortSMA[-1] > longSMA[-1]):
            return 1;
        elif (shortSMA[-1] < longSMA[-1]) :
            return -1;
        else:
            return 0;


    def checkPolinger(self,currentDate,buyInterval,buythreshold,soldInterval,soldthreshold,openArray,closeArray):
        buyStart = max(currentDate - buyInterval, 0);
        buySMA = np.average(closeArray[buyStart:currentDate]) if not buyStart == currentDate else openArray[0];     
        buyalpha = 0;
        for i in range(0,currentDate-buyStart):
            buyalpha = buyalpha + (closeArray[i+buyStart] - buySMA)*(closeArray[i+buyStart] - buySMA);
        
        buyalpha = np.sqrt ( buyalpha / (currentDate-buyStart))  
        

        buyParam = 0;
        if (closeArray[currentDate] > buySMA + buythreshold * buyalpha):
            buyParam = 1;


        soldStart = max(currentDate - soldInterval, 0);
        soldSMA = np.average(closeArray[soldStart:currentDate]) if not soldStart == currentDate else openArray[0];     
        soldalpha = 0;
        for i in range(0,currentDate-soldStart):
            soldalpha = soldalpha + (closeArray[i+soldStart] - soldSMA)*(closeArray[i+soldStart] - soldSMA);
        soldalpha = np.sqrt ( soldalpha / (currentDate-soldStart))

        soldParam = 0;
        if (closeArray[currentDate] < soldSMA - soldthreshold * soldalpha):
            soldParam = 1;

        ## buyParam= 0 and sold Param = 0: 売買する必要がない
        ## buyParam= 1 and sold Param = 1: 売りも買いもするので結局±ゼロ
        if (buyParam == soldParam):
            return 0;
        if (buyParam == 1):
            return 1;
        if (soldParam == 1):
            return -1;

    def calcAllSMA(self,startPoint,lerningInterval,buyLongInterval,buyShortInterval, soldLongInterval, soldShortInterval,openArray ,closeArray):
        
        self.buyLongSMAArray = [0]*len(self.trainData);
        self.buyShortSMAArray = [0]*len(self.trainData);
        self.soldLongSMAArray = [0]*len(self.trainData);
        self.soldShortSMAArray = [0]*len(self.trainData);

        self.buyLongSMAArray[0] = openArray[0];
        self.buyShortSMAArray[0] = openArray[0];
        self.soldLongSMAArray[0] = openArray[0];
        self.soldShortSMAArray[0] = openArray[0];

        for c in range(1,lerningInterval-1+startPoint):
            currentDate = c;
            self.buyLongSMAArray[currentDate]  = (self.calcSMA(currentDate,buyLongInterval,closeArray));
            self.buyShortSMAArray[currentDate]  = (self.calcSMA(currentDate,buyShortInterval,closeArray));
            self.soldLongSMAArray[currentDate]  = (self.calcSMA(currentDate,soldLongInterval,closeArray));
            self.soldShortSMAArray[currentDate]  = (self.calcSMA(currentDate,soldShortInterval,closeArray));
        

    def calcAllEMA(self,startPoint,lerningInterval,buyLongInterval,buyShortInterval, buyPercent,soldLongInterval, soldShortInterval, soldPercent,openArray ,closeArray):
        self.buyLongEMAArray = [0]*len(self.trainData);
        self.buyShortEMAArray = [0]*len(self.trainData);
        self.soldLongEMAArray = [0]*len(self.trainData);
        self.soldShortEMAArray = [0]*len(self.trainData);

        self.buyLongEMAArray[0] = openArray[0]
        self.buyShortEMAArray[0] = openArray[0]
        self.soldLongEMAArray[0] = openArray[0]
        self.soldShortEMAArray[0] = openArray[0]

        for c in range(1,lerningInterval-1 + startPoint):
            currentDate = c;
            self.buyLongEMAArray[currentDate] = self.calcEMA(currentDate,buyLongInterval,buyPercent,self.buyLongSMAArray,closeArray) ;
            self.buyShortEMAArray[currentDate] =  self.calcEMA(currentDate,buyShortInterval,buyPercent,self.buyShortSMAArray,closeArray);
            self.soldLongEMAArray[currentDate] =  self.calcEMA(currentDate,soldLongInterval,soldPercent,self.soldLongSMAArray,closeArray) ;
            self.soldShortEMAArray[currentDate] =  self.calcEMA(currentDate,soldShortInterval,soldPercent,self.soldShortSMAArray,closeArray);

    ## SMAを用いた売買決定 (テクニカル手法の一つ)
    ## currentDate: 日付
    ## 1: buy, -1: sold, 0: none
    ## interval: 10
    def checkBySMA(self, currentDate):
        intarval = 10;
        startPoint = max(currentDate - intarval, 0);

        buyLongSMA      = self.buyLongSMAArray[startPoint:currentDate]
        buyShortSMA     = self.buyShortSMAArray[startPoint:currentDate]        
        soldLongSMA     = self.soldLongSMAArray[startPoint:currentDate]
        soldShortSMA    = self.soldShortSMAArray[startPoint:currentDate]

        checkerBuy = self.checkGoldenOrDead(buyLongSMA,buyShortSMA);
        checkerSold = self.checkGoldenOrDead(soldLongSMA,soldShortSMA);

        ## 売り買い同時にする場合結局行動はしない
        if(checkerBuy == 1 and checkerSold==-1):
            return 0;

        if(checkerBuy == 1):
            return 1;
        if(checkerSold == -1):
            return -1;

        return 0;

    ## SMAを用いた売買決定 (テクニカル手法の一つ)
    ## currentDate: 日付
    ## 1: buy, -1: sold, 0: none
    def checkByEMA(self, currentDate):
        intarval = 10;
        startPoint = max(currentDate - intarval, 0);

        buyLongEMA      = self.buyLongEMAArray[startPoint:currentDate]
        buyShortEMA     = self.buyShortEMAArray[startPoint:currentDate]        
        soldLongEMA     = self.soldLongEMAArray[startPoint:currentDate]
        soldShortEMA    = self.soldShortEMAArray[startPoint:currentDate]

        checkerBuy = self.checkGoldenOrDead(buyLongEMA,buyShortEMA);
        checkerSold = self.checkGoldenOrDead(soldLongEMA,soldShortEMA);

        ## 売り買い同時にする場合結局行動はしない
        if(checkerBuy == 1 and checkerSold==-1):
            return 0;

        if(checkerBuy == 1):
            return 1;
        if(checkerSold == -1):
            return -1;

        return 0;   

    ## テクニカル手法を用いて結局買うかどうかを判定する関数
    def isBuyOrSold(self,whichOne,whichTwo,whichThree,whichfour):
        test = np.sum([whichOne,whichTwo,whichThree,whichfour]);
       # test = np.sum([whichOne,whichTwo]);
        if (test > 0):
            return 1;
        elif (test < 0):
            return -1;
        elif (test == 0):
            return 0;

    def evaluate(self,solution:Solution):
        self.repair(solution);
        money = self.seedMoney;
        currentStack = 0;
        self.calcAllSMA(self.trainStartPoint,self.trainLerningInterval,self.intervals[solution.variables[6]] , self.intervals[solution.variables[7]] , self.intervals[solution.variables[8]], self.intervals[solution.variables[9]], self.trainOpenArray ,self.trainCloseArray);
        self.calcAllEMA(self.trainStartPoint,self.trainLerningInterval,self.intervals[solution.variables[10]] , self.intervals[solution.variables[11]], self.expSmootParam[solution.variables[12]] , self.intervals[solution.variables[13]] , self.intervals[solution.variables[14]], self.expSmootParam[solution.variables[15]] , self.trainOpenArray ,self.trainCloseArray);
        StackList = [];
        Loss = 0;

        for i in range(1,self.trainLerningInterval):
            index = self.trainStartPoint + i;
            ## 購入意思決定判断
            whichForChannel =self.checkByChannelBreakOut(index, self.intervals[solution.variables[0]],self.intervals[solution.variables[1]],self.trainHighArray,self.trainLowArray,self.trainOpenArray);
            whichForSMA =self.checkBySMA(index);
            whichForEMA =self.checkByEMA(index);
            whichForPOL = self.checkPolinger(index,self.intervals[solution.variables[2]],self.coefficients[solution.variables[3]],self.intervals[solution.variables[4]],self.coefficients[solution.variables[5]],  self.trainOpenArray,self.trainCloseArray);

            if ( (self.isBuyOrSold(whichForChannel,whichForSMA,whichForEMA,whichForPOL)==1) and (money > self.unit*self.trainOpenArray[index])):
                money = money - self.unit*self.trainOpenArray[index] - self.Commision;
                currentStack = currentStack + self.unit;
                StackList.append(self.unit*self.trainOpenArray[index]);
            elif( (self.isBuyOrSold(whichForChannel,whichForSMA,whichForEMA,whichForPOL)==-1) and currentStack / self.unit > 0):
                money = money + self.unit*self.trainOpenArray[index] - self.Commision;
                currentStack = currentStack - self.unit;
                profit = self.unit*self.trainOpenArray[index] - StackList.pop(0);
                if(profit < 0):
                    Loss = Loss + profit;
                        
        money = money + currentStack*self.trainCloseArray[-1]- self.Commision;
     
        ## 第一目的が，GAのFitness,　第二目的は利益
        solution.objectives = [money- self.seedMoney,money- self.seedMoney];
#        solution.objectives = [-1*Loss + 0.001 *(money- self.seedMoney)  ,money- self.seedMoney];

    def testTrial(self,solution:Solution):
        money = self.seedMoney;
        currentStack = 0;
        Loss = 0;
        self.calcAllSMA(self.testStartPoint,self.testLerningInterval,self.intervals[solution.variables[6]],self.intervals[solution.variables[7]], self.intervals[solution.variables[8]],self.intervals[solution.variables[9]],self.testOpenArray ,self.testCloseArray);
        self.calcAllEMA(self.testStartPoint,self.testLerningInterval,self.intervals[solution.variables[10]] , self.intervals[solution.variables[11]], self.expSmootParam[solution.variables[12]] , self.intervals[solution.variables[13]] , self.intervals[solution.variables[14]], self.expSmootParam[solution.variables[15]] , self.testOpenArray ,self.testCloseArray);
        for i in range(1,self.testLerningInterval):
            index = self.testStartPoint + i;


            ## 購入意思決定判断
            whichForChannel =self.checkByChannelBreakOut(index, self.intervals[solution.variables[0]],self.intervals[solution.variables[1]],self.testHighArray,self.testLowArray,self.testOpenArray);
            whichForSMA =self.checkBySMA(index);
            whichForEMA =self.checkByEMA(index);
            whichForPOL = self.checkPolinger(index,self.intervals[solution.variables[2]],self.coefficients[solution.variables[3]],self.intervals[solution.variables[4]],self.coefficients[solution.variables[5]],  self.testOpenArray,self.testCloseArray);

            if ( (self.isBuyOrSold(whichForChannel,whichForSMA,whichForEMA,whichForPOL) == 1) and (money > self.unit*self.testOpenArray[index])):
                money = money - self.unit*self.testOpenArray[index] -self.Commision;
                currentStack = currentStack + self.unit;
            elif((self.isBuyOrSold(whichForChannel,whichForSMA,whichForEMA,whichForPOL)==-1) and currentStack / self.unit > 0):
                money = money + self.unit*self.testOpenArray[index]-self.Commision;
                currentStack = currentStack - self.unit;                


        money = money + currentStack*self.testCloseArray[-1]- self.Commision;

        return [money- self.seedMoney,money- self.seedMoney];

    def initialize(self,solution:Solution):
        solution.variables = np.array([random.randint(0,solution.getDivision()[i]) for i in range(0,self.nOfVariables)]);        


if __name__ == "__main__":
    ## data frame ["Open","High","Low","Close","Volume"];
    df = pd.read_csv("SonyData.csv");
    print(df.get("Open").values);


    #try:
    #    stackData = pd.read_csv("SonyData.csv");
    #    problem = stackTradeProblem(stackData,1000000);
    #except Exception as e:
    #    from traceback import print_exe;
    #    print_exe();

