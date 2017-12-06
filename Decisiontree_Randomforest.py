import numpy as np
import pandas as pd
import math
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt # this is used for the plot the graph
from sklearn.model_selection import train_test_split # to split the data into two parts
from sklearn.ensemble import RandomForestRegressor

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_pred - y_true), axis=-1))
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [10, 6]
    train = pd.read_csv("E:\\4thyear\\BigData\\xgb-train.csv")
    test = pd.read_csv("E:\\4thyear\\BigData\\xgb-test.csv")
    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    test['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['pickup_weekday'] = train['pickup_datetime'].dt.weekday
    train['pickup_month'] = train['pickup_datetime'].dt.month
    train['pickup_hour'] = train['pickup_datetime'].dt.hour

    test['pickup_weekday'] = test['pickup_datetime'].dt.weekday
    test['pickup_month'] = test['pickup_datetime'].dt.month
    test['pickup_hour'] = test['pickup_datetime'].dt.hour

    train['night_trip'] = [True if x < 7 else False for x in train['pickup_hour']]
    train['rush_hour'] = [True if 9 < x < 20 else False for x in train['pickup_hour']]
    train['weekday'] = [True if x < 5 else False for x in train['pickup_weekday']]
    test['night_trip'] = [True if x < 7 else False for x in test['pickup_hour']]
    test['rush_hour'] = [True if 9 < x < 20 else False for x in test['pickup_hour']]
    test['weekday'] = [True if x < 5 else False for x in test['pickup_weekday']]
    log_trip_duration = np.log(train['trip_duration'].values + 1)
    train['log_trip_duration'] = log_trip_duration
    #test.to_csv("E:\\4thyear\\BigData\\test-new.csv")
    #train.to_csv("E:\\4thyear\\BigData\\train-new.csv")
    DO_NOT_USE_FOR_TRAINING = [
        'id', 'pickup_datetime', 'jfk_dist_drop','jfk_dist_pick','lg_dist_drop','lg_dist_pick' ,'dropoff_datetime','speed','store_and_fwd_flag', 'trip_duration', 'pickup_date', 'log_trip_duration','date'
    ]
    new_df = train.drop([col for col in DO_NOT_USE_FOR_TRAINING if col in train], axis=1)
    new_df_test = test.drop([col for col in DO_NOT_USE_FOR_TRAINING if col in test], axis=1)

    #new_df['store_and_fwd_flag'] = 1 * new_df['store_and_fwd_flag'] == True
    #new_df_test['store_and_fwd_flag'] = 1 * new_df['store_and_fwd_flag'] == True
    new_df.columns == new_df_test.columns
    y = np.log(train['trip_duration'].values)
    train_attr = np.array(new_df)
    train_attr.shape
    train_x, val_x, train_y, val_y = train_test_split(train_attr, y, test_size=0.2)
    del train_attr
    TREE_REGRESSORS = [
        # These model are not tunned, default params in using
        DecisionTreeRegressor(), RandomForestRegressor(n_estimators=50)
    ]
    models = []
    for regressor in TREE_REGRESSORS:
        clf = regressor
        clf = clf.fit(train_x, train_y)
        models.append(clf)
    for model in models:
        # train_y is logged so rmse computes rmsle
        train_rmsle = rmsle(train_y, model.predict(train_x))
        val_rmsle = rmsle(val_y, model.predict(val_x))
        train_rmse = rmse(train_y, model.predict(train_x))
        val_rmse = rmse(val_y, model.predict(val_x))
        #print('With model: {}\nTrain RMSLE: {}\nVal. RMSLE: {}'.format(model, train_rmsle, val_rmsle))
        print('With model: {}\nTrain RMSLE: {}\nVal. RMSLE: {}'.format(model, train_rmse, val_rmse))
        print "Train"
        print"Variance"
        print explained_variance_score(y_pred=model.predict(train_x), y_true=train_y)
        print "Mean absolute error"
        print mean_absolute_error(y_pred=model.predict(train_x), y_true=train_y)
        print "Mean squared error"
        print mean_squared_error(y_pred=model.predict(train_x), y_true=train_y)
        print "Mean squared log error"
        print mean_squared_log_error(y_pred=model.predict(train_x), y_true=train_y)
        print "r2 score"
        print r2_score(y_pred=model.predict(train_x), y_true=train_y)
        print "Validation"
        print"Variance"
        print explained_variance_score(y_pred=model.predict(val_x), y_true=val_y)
        print "Mean absolute error"
        print mean_absolute_error(y_pred=model.predict(val_x), y_true=val_y)
        print "Mean squared error"
        print mean_squared_error(y_pred=model.predict(val_x), y_true=val_y)
        print "Mean squared log error"
        print mean_squared_log_error(y_pred=model.predict(val_x), y_true=val_y)
        print "r2 score"
        print r2_score(y_pred=model.predict(val_x), y_true=val_y)
    test_attr = np.array(new_df_test)
    model_dt, model_rf = models
    pred_dt = model_dt.predict(test_attr)
    pred_dt = np.exp(pred_dt)-1
    submission = pd.concat([test['id'], pd.DataFrame(pred_dt, columns=['trip_duration'])], axis=1)
    submission.to_csv('submission-dt.csv', index=False)
    importances_dt = model_dt.feature_importances_
    feature_name = new_df.columns
    indices = np.argsort(importances_dt)[::-1]
    # Print the feature ranking
    print("Feature ranking Decision tree:")
    for f in range(train_x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_name[indices[f]], importances_dt[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances Decision Tree")
    plt.bar(range(train_x.shape[1]), importances_dt[indices],
            color="r", align="center")
    plt.xticks(range(train_x.shape[1]), indices)
    plt.xlim([-1, train_x.shape[1]])
    plt.show()
    pred_rf = model_rf.predict(test_attr)
    pred_rf = np.exp(pred_rf)-1
    submission = pd.concat([test['id'], pd.DataFrame(pred_rf, columns=['trip_duration'])], axis=1)
    submission.to_csv('submission-rf.csv', index=False)
    importances_rf = model_dt.feature_importances_
    feature_name = new_df.columns
    indices = np.argsort(importances_rf)[::-1]
    # Print the feature ranking
    print("Feature ranking Random Forest:")
    for f in range(train_x.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, feature_name[indices[f]], importances_rf[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances Random Forest")
    plt.bar(range(train_x.shape[1]), importances_rf[indices],
            color="r", align="center")
    plt.xticks(range(train_x.shape[1]), indices)
    plt.xlim([-1, train_x.shape[1]])
    plt.show()