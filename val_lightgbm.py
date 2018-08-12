
# coding: utf-8

# # lightGBM hyperparameter optmization 

# In[1]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import lightgbm as lgb
import load_datasets
import numpy as np


# In[2]:


X_train, y_train = load_datasets.train(max_date_block_num=27)
X_val, y_val = load_datasets.val(max_date_block_num=30)
D_train = lgb.Dataset(X_train, y_train)
D_val = lgb.Dataset(X_val, y_val)


# In[3]:


def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['objective'] = 'rmse'
    lgb_params['metric'] = 'rmse'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1

    return lgb_params

space ={
        'num_leaves': hp.quniform('num_leaves', 10, 200, 1),
        'min_data_in_leaf':  hp.quniform('min_data_in_leaf', 10, 200, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 64, 512, 1),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 5, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
       }


# In[4]:


obj_call_count = 0
cur_best_loss = np.inf
log_writer = open( './log/lgb-hyperopt-log.txt', 'w' )


# In[5]:


def objective(space):
    global obj_call_count, cur_best_loss

    obj_call_count += 1

    print('\nXGB objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )

    model = lgb.train(lgb_params,
                           D_train,
                           num_boost_round=10000,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=100,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )

    nb_trees = model.best_iteration
    val_loss = model.best_score['valid_0']['rmse']

    print('nb_trees={} val_loss={}'.format(nb_trees, val_loss))
    
    log_writer.write('val_loss={:<7.5f} Params:{} nb_trees={}\n'.format(val_loss, params_str, nb_trees ))
    log_writer.flush()

    if val_loss<cur_best_loss:
        cur_best_loss = val_loss
        print('NEW BEST LOSS={}'.format(cur_best_loss))

    return{'loss':val_loss, 'status': STATUS_OK }


# In[6]:


N_HYPEROPT_PROBES = 20
HYPEROPT_ALGO = tpe.suggest
trials = Trials()
best = fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

log_writer.write('-'*50)
log_writer.write('The best params:')
log_writer.write('{}'.format(best))
log_writer.close()

