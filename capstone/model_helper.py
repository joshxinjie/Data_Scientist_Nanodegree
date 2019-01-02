import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def promotion_strategy_indicator(X_test, model, offer_num):
    """
    DESCRIPTION:
        Returns the promotional strategy, identidying which individuals should
        receive the promotions. Returns promotion array, with 1 indicating the
        customer should receive the promotion and 0 indicating customer should
        not receive the promotion
    
    INPUTS:
        X_test - The demographics and transactional behaviour features
        model - The trained uplift model
        offer_num - The offer id that we are currently analyzing
    OUTPUTS:
        promotion - The array containing the promotion strategy
    """
    offer_name = "offer_id_"+str(offer_num)
    
    # Predict probaility of profit if given offer
    X_test[offer_name] =  1
    preds_treat = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
    
    # Predict probaility of profit if not given an offer
    X_test[offer_name] =  0
    preds_cont = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)
    
    lift = preds_treat[:,1] - preds_cont[:,1]
    
    promotion = np.zeros(X_test.shape[0])
    
    promotion[np.where(lift > 0)] = 1
    
    return promotion

def test_results(offer_num, offer_strategy, test_df):
    """
    DESCRIPTION:
        Returns the results of the promotion strategy. Both IRR
        and NIR will be computed
    
    INPUTS:
        offer_num - The offer id that we are currently analyzing
        offer_strategy - The promotion strategy
        test_df - The test dataframe
    OUTPUTS:
        offer_irr - The IRR of the promotion strategy
        offer_nir - The NIR of the promotion strategy
    """    
    # get subset of test dataframe where we predicted individual should 
    # receive this offer
    pred_yes_df = test_df.iloc[np.where(offer_strategy==1)]
    offer_name = "offer_id_"+str(offer_num)
    
    treat_group = pred_yes_df[pred_yes_df[offer_name]==1]
    ctrl_group = pred_yes_df[pred_yes_df["offer_id_10"]==1]
        
    n_treat = treat_group.shape[0]
    n_ctrl = ctrl_group.shape[0]
        
    n_treat_purch = treat_group['num_trans'].sum()
    n_ctrl_purch =  ctrl_group['num_trans'].sum()
    #n_treat_purch = treat_group['monthly_purchase'].sum()
    #n_ctrl_purch =  ctrl_group['monthly_purchase'].sum()
        
    if n_treat > 0 and n_ctrl > 0:
        offer_irr = n_treat_purch / n_treat - n_ctrl_purch / n_ctrl

        treat_rev = treat_group['profit'].sum()
        crtl_rev = ctrl_group['monthly_amt_spent'].sum()

        offer_nir = treat_rev - crtl_rev

    elif n_treat > 0:
        offer_irr = 0
        treat_rev = treat_group['profit'].sum()
        offer_nir = treat_rev
            
    elif n_ctrl > 0:
        offer_irr = 0
        ctrl_rev = ctrl_group['monthly_amt_spent'].sum()
        offer_nir = 0 - ctrl_rev
            
    else:
        offer_irr = 0
        offer_nir = 0
    
    return offer_irr, offer_nir

def generate_offer_monthly_data(offer_name, monthly_data, start_month=2,\
                                end_month=19):
    """
    DESCRIPTION:
        Generate the subset of the original monthly data that is relevant to 
        the current offer.
    
    INPUTS:
        offer_name - The offer id that we are currently analyzing
        monthly_data - The original monthly data
        start_month - The month from which our data subset should begin in
        end_month - The final month for which our data subset
    
    OUTPUTS:
        new_monthly_data - the subset of the original monthly data relevant to 
        the current offer
    """

    for month_num in range(start_month,end_month+1):
        # get the current month's data
        month_subset = monthly_data[monthly_data['month_num']==month_num]
        month_subset = month_subset[(month_subset[offer_name]==1) |\
                                    (month_subset['offer_id_10']==1)]
        # get individuals who received the offer during the month
        month_offer_indiv =\
        month_subset[month_subset[offer_name]==1].per_id.unique()
        if month_num == start_month:
            new_monthly_data =\
            month_subset[month_subset['per_id'].isin(month_offer_indiv)]
        else:
            new_month_data =\
            month_subset[month_subset['per_id'].isin(month_offer_indiv)]
            new_monthly_data =\
            pd.concat([new_monthly_data, new_month_data], axis=0)
    new_monthly_data.reset_index(inplace=True)
    return new_monthly_data

def do_pca(data, n_components=None):
    '''
    DESCRIPTION:
        Transforms data using PCA to create n_components, and provides back the results of the
        transformation.

    INPUTS:
        data - the data you would like to transform
        n_components - int - the number of principal components to create

    OUTPUTS: 
        pca - the pca object created after fitting the data
        X_pca - the transformed X matrix with new number of components
    '''
    pca = PCA(n_components)
    X_pca = pd.DataFrame(pca.fit_transform(data))
    if n_components != None:
        X_pca.columns = ["pca_comp_" + str(i) for i in range(n_components)]
    X_pca.index = data.index
    
    return pca, X_pca

def scree_plot(pca, n_comp=None):
    '''
    DESCRIPTION:
        Creates a scree plot associated with the principal components 
    
    INPUTS:
        pca - the result of instantian of PCA in scikit learn    
    
    OUTPUTS:
        None
    '''
    if n_comp == None: # If no n_comp is provided, use all components
        num_components = len(pca.explained_variance_ratio_) #n_comp is provided
    elif n_comp < len(pca.explained_variance_ratio_):
        num_components = n_comp
    else: 
        #If the n_comp provided is greater than the total number of components,
        # then use all components
        num_components = len(pca.explained_variance_ratio_)
    indices = np.arange(num_components)
    values = pca.explained_variance_ratio_
    
    values = values[:num_components]
    
    plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    # Create array of cumulative variance explained for each n^th component
    cumulative_values = np.cumsum(values)
    # Plot bar chart of variance explained vs each component
    ax.bar(indices, values, color='tab:red')
    # Plot line chart of cumulative variance explained vs number of components
    ax.plot(indices, cumulative_values)
    
    # Plot the annotations only if there are less than 21 components, 
    # else it gets messy
    if num_components <= 20:
        for i in range(num_components):
            ax.annotate(r"%s%%" % ((str(values[i]*100)[:4])),\
                        (indices[i]+0.2, values[i]), va="bottom",\
                        ha="center", fontsize=12)
        
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    
def grid_search_indicator_pca(offer_num, max_depth_list, upsamp_ratio_list,\
                              min_child_weight_list, X_train, Y_train,\
                              X_valid, Y_valid, X_test, train, valid, test):
    """
    DESCRIPTION:
        Runs a grid search through the given parameters and finds the 
        parameters that produce the best validation NIR
    
    INPUTS:
        offer_num - The offer id that we are currently analyzing
        max_depth_list - The list of maximum depth parameters
        upsamp_ratio_list - The list of oversampling ratio parameters
        min_child_weight_list - The list of minimum child weight parameters
        X_train - The training features
        Y_train - The labels for the training data
        X_valid - The validation features
        Y_valid - The labels for the validation data
        X_test - The test features
        train - The training dataframe
        valid - The validation dataframe
        test - The test dataframe
    OUTPUTS:
        best_depth - The best maximum depth parameter
        best_ratio - The best oversampling ratio
        best_weight - The best minimum child weight parameter
        best_valid_nir - The best validation NIR
        best_test_nir - The test BIR for the optimized strategy
    """
    total_num_models =\
    len(max_depth_list) * len(upsamp_ratio_list) * len(min_child_weight_list)
    cnt = 1
    
    # record parameters for best valid nir
    best_valid_nir = 0
    best_test_nir = 0
    best_depth = 0
    best_ratio = 0
    best_weight = 0
    
    # record parameters for positive valid and test nir
    pos_strat_params = []
    
    offer_name = "offer_id_" + str(offer_num)
    
    for depth in max_depth_list:
        for up_ratio in upsamp_ratio_list:
            for weight in min_child_weight_list:
                X_train_offer = X_train[X_train[offer_name]==1]
                X_train_no_offer = X_train[X_train[offer_name]==0]

                Y_train_offer = Y_train.iloc[np.where(train[offer_name]==1)]
                Y_train_no_offer = Y_train.iloc[np.where(train['offer_id_10']==1)]
                
                X_train_offer.drop(columns=[offer_name], inplace=True)
                
                sm = SMOTE(random_state=42, ratio = up_ratio)
                
                X_train_offer_upsamp, Y_train_offer_upsamp =\
                sm.fit_sample(X_train_offer, Y_train_offer)
                
                X_train_offer_upsamp =\
                pd.DataFrame(X_train_offer_upsamp,\
                             columns=X_train_offer.columns.values)
                
                Y_train_offer_upsamp = pd.Series(Y_train_offer_upsamp)
                
                X_train_offer_upsamp[offer_name] = 1
                
                X_train_upsamp =\
                pd.concat([X_train_offer_upsamp, X_train_no_offer], axis=0)
                
                Y_train_upsamp =\
                pd.concat([Y_train_offer_upsamp, Y_train_no_offer], axis=0)
                
                eval_set = [(X_train_upsamp, Y_train_upsamp), (X_valid, Y_valid)]
                model = xgb.XGBClassifier(learning_rate = 0.1,\
                                          max_depth = depth,\
                                          min_child_weight = weight,\
                                          objective = 'binary:logistic',\
                                          seed = 42,\
                                          gamma = 0.1,\
                                          silent = True)
                model.fit(X_train_upsamp, Y_train_upsamp, eval_set=eval_set,\
                          eval_metric="aucpr", verbose=False,\
                          early_stopping_rounds=20)
                
                valid_promo_strat =\
                promotion_strategy_indicator(X_valid, model, offer_num)
                
                valid_irr, valid_nir =\
                test_results(offer_num, valid_promo_strat, valid)
                
                test_promo_strat =\
                promotion_strategy_indicator(X_test, model, offer_num)
                
                test_irr, test_nir =\
                test_results(offer_num, test_promo_strat, test)
                
                print("Progress: {}/{}, Depth: {}, Ratio: {:.3f}, Weight: {}, Valid NIR: {:.2f}, Test NIR: {:.2f}".format(cnt, total_num_models, depth, up_ratio, weight, valid_nir, test_nir))
                cnt += 1
                
                # record parameters for best validation NIR
                if valid_nir > best_valid_nir:
                    best_valid_nir = valid_nir
                    best_test_nir = test_nir
                    best_depth = depth
                    best_ratio = up_ratio
                    best_weight = weight
                    print("Current Best Depth: {}, Upsampling Ratio: {}, Min Child Weight: {}".format(depth, up_ratio, weight))
                    print("Current Best Valid IRR: {:.2f}, NIR: {:.4f}".format(valid_irr, valid_nir))
                    print("Current Best Test IRR: {:.2f}, NIR: {:.4f}".format(test_irr, test_nir))
                # record parameters that obtain positive valid and test NIR
                if (valid_nir > 0) and (test_nir > 0):
                    pos_strat_params.append((valid_nir, test_nir, depth, up_ratio, weight))
                    
    return best_depth, best_ratio, best_weight, best_valid_nir, best_test_nir, pos_strat_params

def print_pos_strat_params(pos_strat_params):
    for params in pos_strat_params:
        valid_nir = params[0]
        test_nir = params[1]
        depth = params[2]
        up_ratio = params[3]
        weight = params[4]
        print("Valid NIR: {:.2f}, Test NIR: {:.2f}, Tree Depth: {}, Upsampling Ratio: {:.2f}, Min Child Weight: {}".format(valid_nir, test_nir, depth, up_ratio, weight))