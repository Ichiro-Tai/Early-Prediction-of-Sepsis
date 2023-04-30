import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
import time
from sklearn.ensemble import RandomForestClassifier
import joblib
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc, roc_curve


def plot_roc_curve(fpr, tpr, label=None, color=None):
    plt.plot(fpr, tpr, linewidth=2, label=label, color=color)
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')



if __name__ == "__main__":
    accuracy_dict = defaultdict()
    auroc_dict = defaultdict()
    #random forest
    path_processed = "../generated_data/other_alg_file1.csv"
    df = pd.read_csv(path_processed)
    y = df.sepsis2
    x_data = df.drop(columns=['sepsis2'])

    abs_scaler = MaxAbsScaler()
    abs_scaler.fit(x_data)
    scaled_data = abs_scaler.transform(x_data)
    x = pd.DataFrame(scaled_data, columns=x_data.columns)
    x = x.astype('float64')
    x = x.fillna(x.mean())

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.6, random_state = 0)

    for column in x_train.columns:
        if pd.isna(x_train[column].mean()):
            x_train[column].fillna(0, inplace=True)
            x_test[column].fillna(0, inplace=True)

    x_test = x_test.fillna(x_train.mean()) 
    t1 = time.time()
    randForest = RandomForestClassifier(n_estimators = 1000, random_state = 1, verbose = 2)
    randForest.fit(x_train, y_train)
    acc = randForest.score(x_test,y_test)
    t2 = time.time()
    print("Random Forest Accuracy : {:.2f}%".format(acc))
    print("Execution time: {:2f}".format((t2-t1)))
    #joblib.dump(randForest, "./random_forest_model.joblib")
    accuracy_dict['RandomForest'] = acc
    
    randForest_probs = randForest.predict_proba(x_test)[:, 1]
    auroc_randForest = roc_auc_score(y_test, randForest_probs)
    auroc_dict['RandomForest'] = auroc_randForest


    #Regression
    t1 = time.time()
    logReg = LogisticRegression(penalty = 'elasticnet',verbose = 2,solver = 'saga', l1_ratio = 0.5)
    logReg.fit(x_train,y_train)
    myScore = logReg.score(x_test,y_test)
    accuracy_dict['LogisticRegression'] = myScore

    logReg_probs = logReg.predict_proba(x_test)[:, 1]
    auroc_logReg = roc_auc_score(y_test, logReg_probs)
    auroc_dict['LogisticRegression'] = auroc_logReg

    t2 = time.time()
    print("Accuracy for Logistic Regression is: {:.2f}".format(myScore))
    print("Execution time: {:2f}".format((t2-t1)))
    model_filename = "logistic_regression_model.joblib"
    #joblib.dump(logReg, model_filename)
    #y_pred = logReg.predict(x_test)
    #results = pd.DataFrame({"True_Labels": y_test, "Predicted_Labels": y_pred})
    #results.to_csv("logistic_regression_results.csv", index=False)

    #SVM 
    t1 = time.time()
    svm = SVC(random_state = 1,verbose=True, probability=True)
    svm.fit(x_train, y_train)
    acc = svm.score(x_test,y_test) 
    accuracy_dict['SVM'] = acc

    svm_probs = svm.predict_proba(x_test)[:, 1]
    auroc_svm = roc_auc_score(y_test, svm_probs)
    auroc_dict['SVM'] = auroc_svm

    t2 = time.time()
    print("SVC acc is: {:.2f}".format(acc))
    print("Execution time: {:2f}".format((t2-t1)))


    #Gradient Boosting Classifier
    t1 = time.time()
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(x_train, y_train)
    y_pred = gbc.predict(x_test)
    accGradBoosting = accuracy_score(y_test, y_pred)
    accuracy_dict["GradientBoostingClassifier"] = accGradBoosting

    gbc_probs = gbc.predict_proba(x_test)[:, 1]
    auroc_gbc = roc_auc_score(y_test, gbc_probs)
    auroc_dict['GradientBoostingClassifier'] = auroc_gbc
    
    gbc_probs = gbc.predict_proba(x_test)[:, 1]
    auroc_gbc = roc_auc_score(y_test, gbc_probs)
    auroc_dict['GradientBoostingClassifier'] = auroc_gbc
    t2 = time.time()
    print(f"Gradient Boosting Classifier accuracy: {accGradBoosting * 100:.2f}%")
    print("Execution time: {:2f}".format((t2-t1)))

    dataForMainModel = {'fn': 2.0,
        'fp': 6.0,
        'labels': [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        'preds': [-0.039462976, -0.16930237, 0.71616066, -0.30551174, 0.13074422, 0.09741242, 0.20466065, 0.3335008, -0.2629251, -0.14954522, 0.32794634, -0.28727394, -0.11338504, 0.110253476, 0.5984774, 0.8595061, -0.1450935, -0.14703463],
        'tn': 7.0,
        'tp': 3.0}

    #labels = dataForMainModel['labels']
    #preds = dataForMainModel['preds']
    #aurocLSTM = roc_auc_score(labels, preds)
    #auroc_dict['LSTM Model'] = aurocLSTM


    #graph AUROC of all models: 
    models = {'RandomForest': randForest, 'LogisticRegression': logReg, 'SVM': svm, 'GradientBoostingClassifier': gbc}
    colors = {'RandomForest': 'blue', 'LogisticRegression': 'green', 'SVM': 'red', 'GradientBoostingClassifier': 'purple'}

    fig = plt.figure()

    for model_name, model in models.items():
        model_probs = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, model_probs)
        auroc = auc(fpr, tpr)
        plot_roc_curve(fpr, tpr, label=f"{model_name} (AUROC = {auroc:.4f})", color=colors[model_name])

    lstm_labels = dataForMainModel['labels']
    lstm_preds = dataForMainModel['preds']
    fpr_lstm, tpr_lstm, _ = roc_curve(lstm_labels, lstm_preds)
    auroc_lstm = auc(fpr_lstm, tpr_lstm)
    plot_roc_curve(fpr_lstm, tpr_lstm, label=f"LSTM (AUROC = {auroc_lstm:.4f})", color='orange')

    plt.legend(loc="lower right")
    fig.savefig("roc_curves.png", dpi=300)

    plt.show()

    #graph accuracy of all models: 
    '''
    #This is for plotting accuracies:
    models = list(accuracy_dict.keys())
    accuracies = list(accuracy_dict.values())

    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    plt.bar(models, accuracies, color=colors)

    plt.title("Model Accuracies")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)
    plt.yticks([i/10 for i in range(0, 11)], ['{:.0f}%'.format(i * 100) for i in [i/10 for i in range(0, 11)]])
    plt.xticks(fontsize=8)
    plt.savefig("model_accuracies.png", dpi=300, bbox_inches="tight")
    plt.show()
    '''





