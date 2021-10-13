#Forcasting Early Churn customer of SoHo Broadband service

#A. Importing Library and data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    plot_roc_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_curve, auc, plot_roc_curve

df = pd.read_csv('./lst_6m.csv')
#B. Refine Data set
#Drop NA values, Columns
df=df.dropna()
df=df.drop(columns=['chu_typ_g','chu_typ_e','chu_typ_f'])

#cast to string: entr_no
df['entr_no']=df['entr_no'].astype('str')

#set index: entr_no
df = df.set_index('entr_no')

#C. Prepare data set for analysis
# data sampling by under_6m (== y axis)
# 526 cases of under_6m = 1, under sample under_6m == 0 to 526 cases
df_1 = df[df['under_6m']==1]
df_2 = df[df['under_6m']==0].sample(n = 526)

# concat df_1, df_2
dfN = pd.concat([df_1,df_2])

# fix columns after RandomForest model
dfLog = dfN.loc[:,['int_only','cnt_070','cnt_iptv','entr_q1','bld_grnd_fcnt','entr_q2','entr_q4','entr_q3','cont_pr_3',
                   'sohotyp_e','cont_pr_0','cnt_int','ch_bd','cnt_cctv','entr_mon','svc_100','svc_500','cust_kd_id',
                   'entr_thu','cust_kd_biz','ch_cd','ccw_si','puar_bld_cnt_5','puar_bld_cnt_6','entr_wed','entr_fri',
                   'puar_bld_cnt_7','cnt_etc','bld_scls_apt_yn','sohotyp_fb','ccw_gu','bldusg_biz','ch_cc',
                   'puar_bld_cnt_3','ch_bc','svc_1G','bldusg_res','puar_bld_cnt_4','entr_tue','puar_bld_cnt_2',
                   'sohotyp_rt','bldusg_etc','sohotyp_a','flo03_blw_mcls_commcl_yn','bld_scls_sale_biz_yn','svc_lte',
                   'sohotyp_f','sohotyp_b','sohotyp_re','flo03_blw_mcls_job_yn','bld_scls_c1nlf_yn','ccw_gun',
                   'entr_sat','cont_pr_1','puar_bld_cnt_1','bldusg_pub','bld_scls_fnnc_yn','sohotyp_c','sohotyp_t',
                   'sohotyp_p','bld_scls_instt_yn','puar_bld_cnt_0','bld_scls_hspt_yn','bld_scls_rh_yn',
                   'flo03_blw_mcls_mnhs_yn','under_6m']]

# Split to training/testing set using sklearn.train_test_split
# y axis = under_6m
# test_size = 0.5
# export format: data frame
x_train, x_test, y_train, y_test = train_test_split(dfLog.drop('under_6m', axis=1), dfLog['under_6m'], test_size = 0.5, random_state=1)

# D. Fit: Logic Regression

reg = linear_model.LogisticRegression()
reg.fit(x_train, y_train)

for i, j in zip(list(dfLog), reg.coef_[0]):
    print('feature:' + str(i) + '    coef: '+ str(j))

# E. Model Evaluation
#Check classification_report, conf_matrix
print('Prediction Accuracy: ', reg.score(X = x_test, y = y_test))
print(classification_report(y_test, reg.predict(x_test)))
reg.predict_proba(x_test)
conf_matrix = pd.DataFrame(
    confusion_matrix(y_test, reg.predict(x_test)).T,
    index=['Predicted Type 0', 'Predicted  Type 1'],
    columns=['True Type 0', 'True Type 1']
)
conf_matrix

#Check roc curve, lift chart
fpr, tpr, _ = roc_curve(y_test, reg.predict(X = x_test))
roc_auc = auc(fpr, tpr)
print('ROC AUC: %0.2f' % roc_auc)
plot_roc_curve(reg, x_test, y_test)

y2_probas = reg.predict_proba(x_test)
skplt.metrics.plot_lift_curve(y_test, y2_probas)
plt.show()