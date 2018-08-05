CAUTION!!
----------

	These scripts include japanese words as comments. So I think you must erase these comments when you use these scripts.


Contents
----------
----------

	This repotorogy is to optimize stack trader system by genetic algorithm (GA).
	In this trade system, we focus on short-term trades using four technical indexs: simple moving average (SMA), exponesial moving average, bollinger band and channel breakout. Each technical index has single or multiple parameters. In my program, each parameter of each techinical index is optimized by using GA. In chorosome, two types of variables is used, one is for buy stacks, the other is for solding stacks. Therefore, in our methods sum of the number of each technical index is 7, so, a length of chorosome is 14 (7x2). If you have questions about this implementation, please be free to contact to me via an e-mail．
	
	

execute.py
-----------

	Multiple trials are executed in parallel by using multiprocessing module
	each trial is executed if you input  "python GA.py (Trial index number)" in command prompt.

GA.py
-----

	This is a script for genetic algorithm (GA). 
	This script includes a one-max problem for debag of GA.
	
StackProblem.py
----------------

	This script is for the stack problems.	
	

Data
-----

	This directory incudes Stack Data about Sony inc as the lerning and test data.
	
	
