import sys
sys.path.append('..')
import json
import pickle as pk
import numpy as np
import pandas as pd
import time
import Standardization as sz
import uvicorn as uv
import fastapi as fs
import DataModele.DataFeaturing as df
from SVDImplementation import Model






def find_users_rated_items(matrix_user_item, sim_users_uk, item_id):
    """
    Find the users similar on user X rated a items

    Parameters
    ----------
    matrix_user_item: (Dataframe)
        matrix_user_item predicted
    sim_users_uk:
    """
    user_item = matrix_user_item.loc[:, item_id]
    user_item = list(user_item[user_item.loc[:] != 0].index.values) #= [1,3,4] = [3,5,2,1]
    sim = sim_users_uk[user_item[0]]
    for i in user_item:
        if i not in sim_users_uk.keys():
            sim = min(sim_users_uk, key=sim_users_uk.get)
        elif sim > sim_users_uk[i]:
            sim = i
    return sim

def item_recommended(user_item, right_singular_vector, all_user_id, user_id, items_name):
    user_similaire = sz.sim_users_uk(right_singular_vector, all_user_id, user_id)
    start_time = time.perf_counter()
    # print(user_similaire)
    items_fast_recommended = []
    for i in user_similaire:
        for note_user_y, note_user_x, item in zip(list(user_item.loc[i, :]), list(user_item.loc[user_id, :]), list(user_item.columns.values)):
            if note_user_y >= 2.00 and note_user_x <= 3.50:
                if item in items_fast_recommended:
                    continue
                else:
                    items_fast_recommended.append(item)
            # print(f"user = {i} note item : {item} = {note_user_y} et ")
    end_time = time.perf_counter()
    times_total  = end_time - start_time
    return df.getNameProduct(items=items_name, items_id=items_fast_recommended), len(user_similaire), times_total

def model_json():
    model = Model(factor = 50, verbose=False)
    iter_array = [1, 3, 5, 13, 10, 20, 25, 45, 100]
    learning_rate = [0.001, 0.01, 0.00001, 0.1]
    parameters = model.optimize(iter_array, learning_rate=learning_rate, n_factor=__class__, regularization=list, type_optimize=str, verbose=bool)

def pipeline_deploy():
    pass


if __name__ == '__main__':
    model = Model(factor=25, verbose=True)
    iter_array = [1, 3, 5, 14, 10, 25, 100, 50]
    alpha = [0.001, 0.01, 0.0001, 0.02, 0.002,  0.003, 0.012, 0.021]
    hyper_parameters = model.optimize(iter_array, n_factor=None, regularization=[0], learning_rate=alpha, type_optimize='alpha', verbose=True)
    # model.plot_learning_curve(iter_array, train_rmse, test_rmse, train_mse, test_mse)
    model.train(n_iter=hyper_parameters['n_iter'], learning_rate=hyper_parameters['learning_rate'])
    user = int(sys.argv[1])
    data_predicting = model.predict_all()
    data_predicting = pd.DataFrame(data_predicting)
    data_predicting = df.decode_ids(data_predicting, model.data_original_ui)
    items_recommended, user_similaire, run_time = item_recommended(data_predicting, model.item_vect.T.astype("float32"), model.user, user, model.product_name)
    print(f"\t\t\tBest recommendation for user_id = {user}")
    print("-------------------------------------Response-------------------------------------")
    print(items_recommended)
    print("-----------------------------------STATISTIQUE------------------------------------")
    print(f"Data Total: {model.user_item.shape[0] * model.user_item.shape[1]} ratings")
    print(f"User similaire : {user_similaire} founds")
    print(f"Product recommended total : {len(items_recommended) + 1} founds")
    print(f"Hyper-parameters: {hyper_parameters}")
    print(f"Times: {run_time * 5:.4f} seconds")
    # print(model.product_name)
