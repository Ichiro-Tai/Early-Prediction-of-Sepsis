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
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    accuracy_dict = defaultdict()
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
    print("Execution time: {}".format((t2-t1)))
    #joblib.dump(randForest, "./random_forest_model.joblib")
    accuracy_dict['RandomForest'] = acc

    #Regression
    t1 = time.time()
    logReg = LogisticRegression(penalty = 'elasticnet',verbose = 2,solver = 'saga', l1_ratio = 0.5)
    logReg.fit(x_train,y_train)
    myScore = logReg.score(x_test,y_test)
    accuracy_dict['LogisticRegression'] = myScore
    t2 = time.time()
    print("Accuracy for Logistic Regression is: {:.2f}".format(myScore))
    print("Execution time: {}".format((t2-t1)))
    model_filename = "logistic_regression_model.joblib"
    #joblib.dump(logReg, model_filename)
    #y_pred = logReg.predict(x_test)
    #results = pd.DataFrame({"True_Labels": y_test, "Predicted_Labels": y_pred})
    #results.to_csv("logistic_regression_results.csv", index=False)

    #SVM 
    t1 = time.time()
    svm = SVC(random_state = 1,verbose=True)
    svm.fit(x_train, y_train)
    acc = svm.score(x_test,y_test) 
    accuracy_dict['SVM'] = acc
    t2 = time.time()
    print("SVC acc is: {:.2f}".format(acc))
    print("Execution time: {}".format((t2-t1)))


    #Gradient Boosting Classifier
    t1 = time.time()
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbc.fit(x_train, y_train)
    y_pred = gbc.predict(x_test)
    accGradBoosting = accuracy_score(y_test, y_pred)
    accuracy_dict["GradientBoostingClassifier"] = accGradBoosting
    t2 = time.time()
    print(f"Gradient Boosting Classifier accuracy: {accGradBoosting * 100:.2f}%")
    print("Execution time: {}".format((t2-t1)))


    #graph accuracy of all models: 
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





