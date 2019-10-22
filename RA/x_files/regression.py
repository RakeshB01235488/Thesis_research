import pandas as pd
import numpy as np

import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.externals import joblib
from sklearn.metrics import r2_score
from util import mean_absolute_percentage_error, return_top_m_corr


# def self_r2_score(y_true, y_pred):
# 	res = np.sum((y_true - y_pred) ** 2)
# 	tot = np.sum((y_true - y_true.mean()) ** 2)
# 	return 1 - res / tot

def rfr_original(filename = 'data/sample_fire_data.csv'):
	'''
	regression, non_time_serise
	'''
	df = pd.read_csv(filename)
	# df[:10]

	# # remove all NA rows if an NaN is in the row
	# # https://www.kaggle.com/aliendev/example-of-pandas-dropna
	# df = df.dropna(axis=0, how='any')

	# # save df without NaN to csv
	# df.to_csv('data/without_NaN_tmp.csv')

	df_values = df.values
	features = df_values[:,1:]
	values = df_values[:,0]

	# random split data for training (1-frac)*100% and testing frac*100%
	frac = 0.1
	# X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=frac, random_state=42)
	X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=frac)
	# X_train_data = X_train[:,1:]
	# X_train_time = X_train[:,0]
	# X_test_data = X_test[:,1:]
	# X_test_time = X_test[:,0]

	# # scale
	scaler = StandardScaler()
	scaler.fit(X_train)

	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)


	max_depth = 30
	# regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=20,
	                                                          # max_depth=max_depth))

	regr_rf = RandomForestRegressor(n_estimators=20, max_depth=max_depth)

	# y_rfr1 = regr_multirf.fit(X_train,y_train).predict(X_test)
	y_rfr2 = regr_rf.fit(X_train,y_train).predict(X_test)

	# print('y_rfr1 mse:'+str(mean_squared_error(y_test,y_rfr1)))
	print('y_rfr2 mse:'+str(mean_squared_error(y_test,y_rfr2)))
	
	# print('y_rfr1 mape:'+str(mean_absolute_percentage_error(y_test,y_rfr1)))
	print('y_rfr2 mape:'+str(mean_absolute_percentage_error(y_test,y_rfr2)))


	# print('y_rfr1 r2:'+str(r2_score(y_test, y_rfr1)))
	print('y_rfr2 r2:'+str(r2_score(y_test, y_rfr2)))


	# # vis results
	# X_id = list(range(len(X_test)))
	# # lw = 2
	# plt.scatter(X_id, y_test, color='darkorange', label='data')
	# plt.scatter(X_id, y_rbf, color='navy', label='RBF model')
	# plt.scatter(X_id, y_lin, color='c', label='Linear model')
	# # plt.scatter(X_test_time, y_poly, color='cornflowerblue', label='Polynomial model')
	# plt.xlabel('id')
	# plt.ylabel('data')
	# plt.title('Support Vector Regression')
	# plt.legend()
	# plt.show()


	# # save model
	# from sklearn.externals import joblib
	# joblib.dump(y_rbf, 'model/y_rbf.joblib')
	# joblib.dump(y_lin, 'model/y_lin.joblib')
	# joblib.dump(y_sig, 'model/y_sig.joblib')
	# # joblib.dump(y_poly, 'model/y_poly.joblib')

def rfr_transformation(filename = 'data/sample_fire_data.csv'):
	'''
	'''
	df = pd.read_csv(filename)
	# df[:10]

	# remove all NA rows if an NaN is in the row
	# https://www.kaggle.com/aliendev/example-of-pandas-dropna
	df = df.dropna(axis=0, how='any')
	# transformation
	df['Area'] = np.log(df['Area'])
	transformed_file = 'data/transformed_data.csv'
	df.to_csv(transformed_file, index=False)
	rfr_original(transformed_file)
 
	

def vis(filename = 'data/sample_fire_data.csv'):
	'''
	'''
	# scatter plot
	
	df = pd.read_csv(filename)
	# scatter_matrix(data)
	# plt.show()

	# correlation graph
	# TODO automatically get name
	# names = ['Mass','Area','PanRowDistanceft','PanColumnDistance','TotalDistance']
	names = ['Mass','log_Area','log_PanRow','log_PanColumn','log_TotalDistance']
	df['Area'] = np.log(df['Area'])
	df['PanRowDistanceft'] = np.log(df['PanRowDistanceft'])
	df['PanColumnDistance'] = np.log(df['PanColumnDistance'])
	df['TotalDistance'] = np.log(df['TotalDistance'])

	correlations = df.corr()
	# plot correlation matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(correlations, vmin=-1, vmax=1)
	fig.colorbar(cax)
	ticks = np.arange(0,5,1)
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(names)
	ax.set_yticklabels(names)
	plt.show()


def load_model():
	'''
	'''
	filename = 'data/randall_data_modify_NH4.csv'
	df = pd.read_csv(filename)
	# df[:10]

	# remove all NA rows if an NaN is in the row
	# https://www.kaggle.com/aliendev/example-of-pandas-dropna
	df = df.dropna(axis=0, how='any')

	
	df_values = df.values
	features = df_values[:,1:]
	values = df_values[:,0]

	# random split data for training (1-frac)*100% and testing frac*100%
	frac = 0.005
	X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=frac, random_state=42)
	X_train_data = X_train[:,1:]
	X_train_time = X_train[:,0]
	X_test_data = X_test[:,1:]
	X_test_time = X_test[:,0]


	nh4_model_rbf = joblib.load('model/y_rbf_NH4.joblib')
	nh4_model_lin = joblib.load('model/y_lin_NH4.joblib')
	nh4_model_sig = joblib.load('model/y_sig_NH4.joblib')

	y_rbf = nh4_model_rbf.predict(X_test_data)
	y_lin = nh4_model_lin.predict(X_test_data)
	y_sig = nh4_model_sig.predict(X_test_data)

	print('rbf mse:'+str(mean_absolute_percentage_error(y_test,y_rbf)))
	print('lin mse:'+str(mean_absolute_percentage_error(y_test,y_lin)))
	print('sig mse:'+str(mean_absolute_percentage_error(y_test,y_sig)))

	# NO3
	filename = 'data/randall_data_modify.csv'
	df = pd.read_csv(filename)
	# df[:10]

	# remove all NA rows if an NaN is in the row
	# https://www.kaggle.com/aliendev/example-of-pandas-dropna
	df = df.dropna(axis=0, how='any')


	df_values = df.values
	features = df_values[:,1:]
	values = df_values[:,0]

	# random split data for training (1-frac)*100% and testing frac*100%
	frac = 0.005
	X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=frac, random_state=42)
	X_train_data = X_train[:,1:]
	X_train_time = X_train[:,0]
	X_test_data = X_test[:,1:]
	X_test_time = X_test[:,0]

	no3_model_rbf = joblib.load('model/y_rbf.joblib')
	no3_model_lin = joblib.load('model/y_lin.joblib')
	no3_model_sig = joblib.load('model/y_sig.joblib')

	y_rbf1 = no3_model_rbf.predict(X_test_data)
	y_lin1 = no3_model_lin.predict(X_test_data)
	y_sig1 = no3_model_sig.predict(X_test_data)

	print('rbf mse:'+str(mean_absolute_percentage_error(y_test,y_rbf1)))
	print('lin mse:'+str(mean_absolute_percentage_error(y_test,y_lin1)))
	print('sig mse:'+str(mean_absolute_percentage_error(y_test,y_sig1)))

def cross_valid_rbf(filename = 'data/sample_fire_data.csv', col_name = 'Mass', first_m_per_corr = 1):
	'''
	'''
	df = pd.read_csv(filename)

	df = df.dropna(axis=0, how='any')

	# 
	col_num = df.shape[1]
	m = int(col_num*first_m_per_corr)
	return_top_m_corr(df, m, col_name)

	df_values = df.values


	features = df_values[:,1:]
	values = df_values[:,0]

	# # normalization between 0 and 1
	# features = features/features.max(axis=0)
	# # print(features[2][1])
	# values = values / values.max(axis=0)

	best_c_rbf = 0
	best_e_rbf = 0
	best_r_2_rbf = 0
	best_rmse_rfr = 100
	best_mape_rbf = 100
	# best_c_sig = 0
	# best_e_sig = 0
	# best_r_2_sig = 0

	fold = 10

	C = 1
	epsilon = 1
	for c_count in range(10):
		# c_count is used for the number of trees
		for e_count in range(10):
			# e_count is used for the max_depth
			C_tmp = C + 5*c_count
			# C_tmp = 10
			e_tmp = epsilon + 5*e_count
			# e_tmp = 0.3

			total_rmse_rfr = 0.0
			total_mape = 0.0
			total_r_2_rbf = 0.0
			# total_error_sig = 0.0
			# total_r_2_sig = 0.0
			for i in range(fold):
				# print('fold {}, c_count {}, e_count {}'.format(i, c_count, e_count))
				# random split data for training (1-frac)*100% and testing frac*100%
				frac = 0.10
				X_train, X_test, y_train, y_test = train_test_split(features, values, test_size=frac)

				# scale
				scaler = StandardScaler()
				scaler.fit(X_train)

				X_train = scaler.transform(X_train)
				X_test = scaler.transform(X_test)


				max_depth = e_tmp
				# regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=20,
				                                                          # max_depth=max_depth))

				regr_rf = RandomForestRegressor(n_estimators=C_tmp, max_depth=max_depth)

				# y_rfr1 = regr_multirf.fit(X_train,y_train).predict(X_test)
				y_rfr2 = regr_rf.fit(X_train,y_train).predict(X_test)


				total_r_2_rbf = r2_score(y_test, y_rfr2) + total_r_2_rbf
				# total_r_2_sig = r2_score(y_test, y_sig) + total_r_2_sig

				total_rmse_rfr = math.sqrt(mean_squared_error(y_test,y_rfr2)) + total_rmse_rfr
				total_mape = mean_absolute_percentage_error(y_test,y_rfr2) + total_mape


			
			if total_r_2_rbf/fold > 0.4:
				print('rfr r_2:'+str(total_r_2_rbf/fold)+", for number of trees="+str(C_tmp)+", and for max_depth="+str(e_tmp))
				if best_r_2_rbf < total_r_2_rbf/fold:
					best_c_rbf = C_tmp
					best_e_rbf = e_tmp
					best_r_2_rbf = total_r_2_rbf/fold
			
		
			if best_rmse_rfr > total_rmse_rfr/fold:
				# best_c_rbf = C_tmp
				# best_e_rbf = e_tmp
				best_rmse_rfr = total_rmse_rfr/fold
			
			if best_mape_rbf > total_mape/fold:
				best_mape_rbf = total_mape/fold

			# if total_r_2_sig/fold > 0.4:
			# 	print('sig r_2:'+str(total_r_2_sig/fold)+", for C="+str(C_tmp)+", and for epsilon="+str(e_tmp))
			# 	if best_r_2_sig < total_r_2_sig/fold:
			# 		best_c_sig = C_tmp
			# 		best_e_sig = e_tmp
			# 		best_r_2_sig = total_r_2_sig/fold
				

	print('-------------------final result-----------------------')
	print('best rbf r_2:'+str(best_r_2_rbf)+", for C="+str(best_c_rbf)+", and for epsilon="+str(best_e_rbf))
	print('best rbf rmse:'+str(best_rmse_rfr))
	print('best rbf mape:'+str(best_mape_rbf))
	# print('best sig r_2:'+str(best_r_2_sig)+", for C="+str(best_c_sig)+", and for epsilon="+str(best_e_sig))
			




# rfr_original()
# rfr_transformation()
cross_valid_rbf()
# vis()