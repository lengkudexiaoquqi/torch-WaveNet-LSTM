BaseLine_AQP 单变量空气质量预测
	BaseLine.py        pytorch构建的基准模型BPNN\RNN\LSTM网络结构
		 NoiseSelect.py     SSA降噪 三个模型的对比程序
		window_select_bpnn.py 滑动窗口选择--BP模型
		window_select_rnn.py 滑动窗口选择--RNN模型
		window_select_lstm.py 滑动窗口选择--LSTM模型
		Single_bp.py   单变量单bp预测

data 数据目录
	combine_res.csv 是 原始数据，后面添加了风向数据
	combine_filter\ssa_10*.csv    是降噪后的数据 其中降噪使用的是matlab里面的SSA代码，具体参见老师以前的代码，				ssa_10.csv是 100个分量，取10个分量的重构时间序列.

	
Diebold-Mariano-Test-master  DM测试，注（文件夹中后缀为npy是预测值，后缀为pkl是保存的模型）
	real_value 是保存真实值 
	muti_factor_predict_value 是多元时序的预测值
	WaveNet-LSTM是WaveNet-LSTM 的预测值和对应的pkl模型文件保存

	dm_test.py 是 dm测试的工具包，使用时导入即可，具体参数详见注释
	main.py  是用来计算多变量和Wavenet之间模型的一个dm测试

MVT_AQP 是融合气象数据的空气质量预测
	BaseLine_MVT 是 多元BP\RNN\LSM的模型结构
	GrangerTest.py 是格兰杰因果检验的代码
	Hyperopt_BPNN.py 是用来对多元BP预测空气质量调参的，主要使用的调参工具 optuna
	Hyperopt_RNN.py 是针对多元RNN预测空气质量调参的
	Hyperopt_LSTM.py 是针对多元LSTM预测空气质量调参的
	test_BPNN.py 是 将调参的参数带入进行bp预测
	test_RNN.py是 将调参的参数带入进行rnn预测
	test_LSTM.py是 将调参的参数带入进行LSTM预测

WaveNet_LSTM_AQP  WaveNet-LSTM空气质量预测
	DE-WaveNet-LSTM.py 是用 差分进化算法对WaveNet-LSTM调参的代码
	Hyperopt-WaveNet-LSTM.py 是用optuna对WaveNet-LSTM进行调参的代码
	load_model_and_predict.py 是加载pkl文件然后预测的代码
	Single_WaveNet_LSTM.py 是单个的WaveNet_LSTM预测的代码
	WaveNet_LSTM.py 是 使用pytorch构建WaveNet_LSTM的网络结构代码

get_data.py 主要是用来得到划分后的训练集和测试集

logger_res.py 是用来打印输出日志，比如调参过程的日志

seed_set.py 是 种子设置的代码




	


	




