import os
import matplotlib.pyplot as plt
from makedata import get_data
from z_score import make_z_score_data
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model


path = r"data_path"
model = load_model(os.path.join(path, "best_model.h5"), compile=False)

from keras.utils.vis_utils import plot_model
# plot_model(model, to_file=os.path.join(path, "model.png"), show_shapes=True, dpi=300)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from makedata import find_data
Dir, filename, label, cc, Epoch = find_data()

# ---------------------------------------------------
for i in range(len(filename)):
    exec("x_%s,y_%s,ID_%s = get_data(Dir,filename[%d],label,cc)" % (filename[i], filename[i], filename[i], i), globals())
    if i == 0:
        exec("zscoreX = x_%s" % (filename[i]), globals())
        global zscoreX
        zscoreX = zscoreX.values
    exec("x_%s = x_%s.astype('float32')" % (filename[i], filename[i]), globals())

# Z_score
from pandas import DataFrame
for i in range(len(filename)):
    exec("data = x_%s" % (filename[i]), globals())
    exec("afterdata = make_z_score_data(zscoreX, data)", globals())
    exec("x_%s = np.around(afterdata, decimals=3)" % (filename[i]), globals())
    exec("x_%s = DataFrame(x_%s,columns=data.columns)" % (filename[i], filename[i]), globals())

# ---------------------------------------------------
if not os.path.exists(os.path.join(path, "ROC")):
    os.makedirs(os.path.join(path, "ROC"))
if not os.path.exists(os.path.join(path, "Proba")):
    os.makedirs(os.path.join(path, "Proba"))
if not os.path.exists(os.path.join(path, "foreplot")):
    os.makedirs(os.path.join(path, "foreplot"))

from plot_ROC import plot_ROC
for i in range(len(filename)):
    exec("global y_%s" % (filename[i]), globals())
    exec("y_true = y_%s" % (filename[i]), globals())
    exec("y_pred = model.predict(x_%s)" % (filename[i]), globals())
    exec("ID = ID_%s" % (filename[i]), globals())
    # plot_ROC(y_true, y_pred, filename[i], i, os.path.join(path,'ROC'))
    y_true = pd.DataFrame(y_true, columns=["label"])
    y_pred = pd.DataFrame(y_pred, columns=["proba"])
    list_proba = pd.concat([ID, y_true, y_pred],axis = 1)
    exec("list_proba.to_csv(os.path.join(path,'Proba','proba_%s.csv'),encoding='gbk', index=None)" % (filename[i]), globals())
    plt.close()




# ======================================dataset============================================
shapnum = 0
print('dataset: ', filename[shapnum])
exec("ID = ID_%s" % (filename[shapnum]), globals())
exec("shapdata = x_%s" % (filename[shapnum]), globals())
exec("shapdatay = y_%s" % (filename[shapnum]), globals())


if not os.path.exists(os.path.join(path, filename[shapnum])):
    os.makedirs(os.path.join(path, filename[shapnum]))
    print('Save path:', os.path.join(path, filename[shapnum]))

# ------------------------------------- Permutation-SHAP -------------------------------------------
def call_model(shapdata):
    return model.predict(shapdata)
# ---------------------------- SHAP ----------------------------
import shap
# shap.explainers._permutation.PermutationExplainer
perm_explainer = shap.Explainer(call_model, shapdata)
# perm_shap_values = perm_explainer(shapdata)
np.savetxt(os.path.join(path,'SHAP.csv'), perm_shap_values.values, delimiter="," )

# =================================================================
myfig = plt.gcf()
shap.summary_plot(perm_shap_values, shapdata, max_display=shapdata.shape[1])
myfig.savefig(os.path.join(path, filename[shapnum], filename[shapnum] + ' SHAP_Beeswarm2.pdf'), dpi=300, bbox_inches='tight')
plt.close()

# # ------------------------------------- KernelExplainer-SHAP -------------------------------------------
explainer = shap.KernelExplainer(model.predict, shap.sample(shapdata))
shap.initjs()
shap.summary_plot(perm_shap_values, shapdata)

# ------------------------------- SHAP-force-plot ------------------------------------
for i in range(min(shapdata.shape[0], 50)):
    print(ID[i], ':', i/min(shapdata.shape[0], 50))
    shap_values = explainer.shap_values(shapdata.iloc[i,:], nsamples = shapdata.shape[0])
    explainer.expected_value = np.around(explainer.expected_value, decimals=3)
    shap.force_plot(explainer.expected_value, np.around(shap_values[0], decimals=3), np.around(shapdata.iloc[i,:], decimals=3), figsize=(30,3), show=False, matplotlib=True, text_rotation=8.5)  # explainer.expected_value,
    plt.savefig(os.path.join(os.path.join(path,"foreplot"), "Group={}, ID={}, label={}, pred={}.tiff".format(filename[shapnum],ID[i],shapdatay.iloc[i],np.around(model.predict(shapdata)[i],decimals=3))),dpi = 300,bbox_inches = 'tight')
    plt.savefig(os.path.join(os.path.join(path,"foreplot"), "Group={}, ID={}, label={}, pred={}.pdf".format(filename[shapnum],ID[i],shapdatay.iloc[i],np.around(model.predict(shapdata)[i],decimals=3))),dpi = 300,bbox_inches = 'tight')

# print('finish')
