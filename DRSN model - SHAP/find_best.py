from makedata import get_data, find_data
from z_score import make_z_score_data
from themodel import trainmodel
import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

targetname = ["train", 0.8, "test", 0.75, "val", 0.75]  # if more than the target AUC, save model

import time
save_filename = time.strftime("Model_%Y-%m-%d_%H.%M.%S", time.localtime(time.time()))
Dir, filename, label, cc, Epoch = find_data()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
for i in range(len(filename)):
    exec("x_%s, y_%s, ID_%s = get_data(Dir, filename[%d], label, cc)" % (filename[i], filename[i], filename[i], i), globals())
    if i == 0:
        exec("zscoreX = x_%s" % (filename[i]), globals())
        global zscoreX
        zscoreX = zscoreX.values
    exec("x_%s = x_%s.astype('float32')" % (filename[i], filename[i]), globals())


from pandas import DataFrame
for i in range(len(filename)):
    exec("data = x_%s" % (filename[i]), globals())
    exec("afterdata = make_z_score_data(zscoreX, data)", globals())
    exec("x_%s = afterdata" % (filename[i]), globals())
    #exec("x_%s = DataFrame(x_%s)" % (filename[i], filename[i]), globals())
    exec("x_%s = DataFrame(x_%s, columns=data.columns)" % (filename[i], filename[i]), globals())

# 准备训练和验证数据
exec("input_size = x_%s.shape[1]" % (filename[0]), globals())  # input_size
exec("batch_size = x_%s.shape[0]" % (filename[0]), globals())  # batch_size
exec("x_testt = x_%s" % (filename[1]), globals())
exec("y_testt = y_%s" % (filename[1]), globals())
exec("x_trainn = x_%s" % (filename[0]), globals())
exec("y_trainn = y_%s" % (filename[0]), globals())
if True:
    list_ac = []
    list_lr = []
    list_hidden = []
    list_seed = []
    listauc_all = pd.DataFrame()
    finallist: DataFrame = pd.DataFrame()
    for i in range(len(filename)):  # 计算auc
        exec("listauc_%s = []" % (filename[i]), globals())
    if not os.path.exists(os.path.join(Dir, save_filename)):
        os.makedirs(os.path.join(Dir, save_filename))

    for seed in range(1, 100, 1):
        print(' ----------------- seed:', seed, '------------------------')
        batch_size = x_trainn.shape[0]
        for ac in ["relu"]: # , "selu", "elu", "softplus", "softsign", "tanh"
            for lr in [0.0005, 0.00065, 0.00055]:
                for hidden_size in [12, 16, 20]:
                    history, model = trainmodel(x_trainn, y_trainn, x_testt, y_testt, lr, hidden_size, Epoch, batch_size, ac)
                    for i in range(len(filename)):
                        exec("auc_%s = roc_auc_score(y_%s, model.predict(x_%s))" % (filename[i], filename[i], filename[i]), globals())
                        exec("'%sAUC:{}'.format(auc_%s)" % (filename[i], filename[i]))
                        exec("listauc_%s.append(auc_%s)" % (filename[i], filename[i]), globals())

                    list_seed.append(seed)
                    list_ac.append(ac)
                    list_lr.append(lr)
                    list_hidden.append(hidden_size)

                    aucta = 0
                    for j in range(len(targetname) // 2):
                        exec("aucta_%s = 0" % (targetname[2 * j]), globals())
                        exec("aucmiddel = %s" % (targetname[2 * j + 1]), globals())
                        exec("if auc_%s < aucmiddel:"
                             "aucta_%s = 1" % (targetname[2 * j], targetname[2 * j]), globals())
                        exec("aucta = aucta + aucta_%s" % (targetname[2 * j]), globals())
                    exec("partauc = auc_%s - auc_%s" % (targetname[0], targetname[2]), globals())
                    if aucta == 0:
                        if partauc < 0.15:
                            if not os.path.exists(os.path.join(Dir,save_filename, "best_model, Epoch{}, seed{}, act{}, lr{}, hidden{}.h5".format(Epoch, seed, ac, lr, hidden_size))):  # 如果路径不存在
                                os.path.exists(os.path.join(Dir,save_filename, "best_model, Epoch{}, seed{}, act{}, lr{}, hidden{}.h5".format(Epoch, seed, ac, lr, hidden_size)))
                            thepath = os.path.join(Dir, save_filename, "best_model, Epoch{}, seed{}, act{}, lr{}, hidden{}".format(Epoch, seed, ac, lr, hidden_size))  # 该模型的路径
                            print('thepath', thepath)
                            if not os.path.exists(os.path.join(thepath, "plot")):
                                os.makedirs(os.path.join(thepath, "plot"))
                            model.save(os.path.join(thepath, "best_model.h5"))
                            history = history[0]
                            epochs = range(1, len(history.history['loss']) + 1)
                            for i in history.history:
                                for j in history.history:
                                    if i[0:3] != "val" and j[0:3] == "val":
                                        if i[-3:] == j[-3:]:
                                            dataframe = pd.DataFrame({"epochs": epochs, i: history.history[i], j: history.history[j]})
                                            dataframe.to_csv(os.path.join(thepath, "plot", i + ".csv"), encoding="gbk", index=False)
                                            plt.plot(epochs, history.history[i], color="aqua", label=i, lw=3)
                                            plt.plot(epochs, history.history[j], color="lime", label="Test_" + i, lw=3)
                                            plt.title('Training and Test')
                                            plt.xlabel('Epochs')
                                            plt.ylabel(i)
                                            plt.legend()
                                            plt.savefig(os.path.join(thepath, "plot", i + ".tiff"), dpi=300)
                                            plt.close()


                    list_seed_ = pd.DataFrame(list_seed, columns=["seed"])
                    # list_ac_ = pd.DataFrame(list_ac, columns=["activation"])
                    list_lr_ = pd.DataFrame(list_lr, columns=["learning rate"])
                    list_hidden_ = pd.DataFrame(list_hidden, columns=["hidden size"])
                    finallist = pd.concat([list_seed_, list_lr_, list_hidden_], axis=1)

                    listauc_all_ = pd.DataFrame()
                    for i in range(len(filename)):
                        exec("listauc__%s= pd.DataFrame(listauc_%s, columns=['auc_%s'])" % (filename[i], filename[i], filename[i]), globals())
                        exec("listauc_all_=pd.concat([listauc_all_,listauc__%s], axis=1)"% (filename[i]), globals())
                    finallist_ = pd.concat([finallist, round(listauc_all_, 3)], axis=1)
                    print(finallist_)
                    finallist_.to_csv(os.path.join(Dir, save_filename, 'AAUC.csv'), encoding="gbk", index=False)

print('finish')