import pandas as pd

# random over sampling for binary classification
def ros(X_train, y_train):
    train_set = pd.concat([X_train, y_train], axis=1)
    count_0, count_1 = train_set.Label.value_counts()
    data_0 = train_set[train_set['Label'] == 0]
    data_1 = train_set[train_set['Label'] == 1]

    data_1_over = data_1.sample(count_0, replace=True)
    data_over = pd.concat([data_0, data_1_over], axis=0)

    print(data_over.Label.value_counts())
    return data_over

def rus(X_train, y_train):
    train_set = pd.concat([X_train, y_train], axis=1)
    count_0, count_1 = train_set.Label.value_counts()
    data_0 = train_set[train_set['Label'] == 0]
    data_1 = train_set[train_set['Label'] == 1]

    data_0_under = data_0.sample(count_1, replace=True)
    data_under = pd.concat([data_1, data_0_under], axis=0)

    print(data_under.Label.value_counts())
    return data_under