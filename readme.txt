BaseLine_AQP Univariate Air Quality Prediction
	Subfolder:  BaseLine.py pytorch constructed benchmark model BPNN\RNN\LSTM network structure
                NoiseSelect.py SSA noise reduction Comparison program for three models
                window_select_bpnn.py Sliding window selection - BP model
                window_select_rnn.py Slide window to select - RNN model
                window_select_lstm.py Sliding window select - LSTM model
                Single_bp.py Univariate single bp prediction


data Data catalog
	combine_res.csv is the original data, and the wind direction data is added later.
	combine_filter\ssa_10*.csv is the data after noise reduction, which noise reduction using the SSA code in matlab, see the teacher's previous code, ssa_10.csv is a reconstructed time series of 100 components, taking 10 components.

	
Diebold-Mariano-Test-master DM test, note (folder with npy suffix is predicted values, pkl suffix is saved model)
	real_value saves the real value 
	muti_factor_predict_value is the predicted value of the multivariate time series.
	WaveNet-LSTM is the predicted value of WaveNet-LSTM and the corresponding pkl model file.
	dm_test.py is the toolkit for dm test, just import it when you use it, see comments for specific parameters.
	main.py is a dm test used to compute the model between multivariate and Wavenet.

MVT_AQP is an air quality prediction that incorporates meteorological data.
	BaseLine_MVT is the model structure of multivariate BP\RNN\LSM.
	GrangerTest.py is the code for Granger causality test.
	Hyperopt_BPNN.py is used to parameterize the multivariate BP air quality prediction, mainly using the parameterization tool optuna.
	Hyperopt_RNN.py is the code for multivariate RNN prediction of air quality.
	Hyperopt_LSTM.py is used to predict air quality by multivariate LSTM.
	test_BPNN.py is to bring in the parameters of the tuning parameter for bp prediction.
	test_RNN.py is to bring in the parameters of the tuning parameter for rnn prediction.
	test_LSTM.py is to bring in the parameters of the tuning parameter for LSTM prediction.

WaveNet_LSTM_AQP WaveNet-LSTM Air Quality Prediction
	DE-WaveNet-LSTM.py is the code to parameterize WaveNet-LSTM with differential evolution algorithm.
	Hyperopt-WaveNet-LSTM.py is the code to parameterize WaveNet-LSTM with optuna.
	load_model_and_predict.py is the code to load the pkl file and then predict.
	Single_WaveNet_LSTM.py is the code for single WaveNet_LSTM prediction.
	WaveNet_LSTM.py is the code to build the network structure of WaveNet_LSTM using pytorch.

    get_data.py is mainly used to get the training set and test set after division.

    logger_res.py is used to print out logs, such as the log of the parameterization process.

    seed_set.py is the code for seed setting.



	


	




