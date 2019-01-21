import h2o
from data_pre import data_pre
from h2o.automl import H2OAutoML
h2o.init(max_mem_size='16G')

train = h2o.upload_file('train.csv')
test = h2o.upload_file('test.csv')

feature_name = [i for i in train.columns if i not in ['样本id','收率']]
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)
x = feature_name
y = '收率'
aml = H2OAutoML(max_models=320, seed=2019, max_runtime_secs=12800)
aml.train(x=feature_name, y=y, training_frame=train)
lb = aml.leaderboard
lb.head(rows=lb.nrows)
automl_predictions = aml.predict(test).as_data_frame().values.flatten()
