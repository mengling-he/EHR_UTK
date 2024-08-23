import pandas as pd
import numpy as np
import statistics
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras as ks
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
import scipy.stats as st
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------#

class Autoencoder(ks.models.Model):
    def __init__(self, actual_dim, latent_dim, activation, loss, optimizer):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = ks.Sequential([
        ks.layers.Flatten(),
        ks.layers.Dense(latent_dim, activation=activation),
        ])

        self.decoder = ks.Sequential([
        ks.layers.Dense(actual_dim, activation=activation),
        ])

        self.compile(loss=loss, optimizer=optimizer, metrics=[ks.metrics.BinaryAccuracy(name='accuracy')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_AE(actual_dim=1, latent_dim=100, activation='relu', loss='MAE', optimizer='Adam'):
    return Autoencoder(actual_dim, latent_dim, activation, loss, optimizer)

# --------------------------------------------------------------------------------------------------#

def run_PCA(X_train_scaled, X_test_scaled, n_components=0.85):
    pca_model = PCA(n_components=n_components)
    pca_model.fit(X_train_scaled)

    PCA_train = pca_model.transform(X_train_scaled)
    PCA_test = pca_model.transform(X_test_scaled)

    return PCA_train, PCA_test, pca_model

# --------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, param_grid = None):
    if param_grid == None:
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

    search = GridSearchCV(estimator = Lasso(),
                          param_grid = param_grid,
                          cv = 5,
                          scoring="neg_mean_squared_error",
                          verbose=0
                          )

    search.fit(X_train_scaled, y_train)
    coefficients = search.best_estimator_.coef_
    importance = np.abs(coefficients)
    remove = np.array(X_train_scaled.columns)[importance == 0]

    LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
    LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]

    return LASSO_train, LASSO_test

# --------------------------------------------------------------------------------------------------#

def run_tSNE(X_train_scaled, X_test_scaled, n_components=2):
    tsne = TSNE(n_components=n_components)
    tSNE_train = tsne.fit_transform(X_train_scaled)
    tSNE_test = tsne.fit_transform(X_test_scaled)

    return tSNE_train, tSNE_test

# --------------------------------------------------------------------------------------------------#

def run_UMAP(X_train_scaled, X_test_scaled):
    pass

# --------------------------------------------------------------------------------------------------#

def run_AE(X_train_scaled, X_test_scaled, param_grid=None):

    if param_grid == None:
        param_grid = {
            'actual_dim' : [len(X_train_scaled.columns)],
            'latent_dim' : [10, 25, 50, 100],
            'activation' : ['relu', 'sigmoid', 'tanh'],
            'loss' : ['MAE', 'binary_crossentropy'],
            'optimizer' : ['SGD', 'Adam']
        }

    model = KerasClassifier(build_fn=create_AE, epochs=10, verbose=0)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=0
    )

    result = grid.fit(X_train_scaled, X_train_scaled, validation_data=(X_test_scaled, X_test_scaled))
    params = grid.best_params_
    autoencoder = create_AE(**params)

    try:
        encoder_layer = autoencoder.encoder
    except:
        exit

    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train.add_prefix('feature_')
    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test.add_prefix('feature_')

    return AE_train, AE_test

# --------------------------------------------------------------------------------------------------#

def run_LASSO2(X_train_scaled, X_test_scaled, y_train, y_test, study=None, param_grid=None, random_state=1992):###872510-1992
    lasso = LogisticRegression(penalty='l1', solver='liblinear', random_state=random_state)
    model = BaggingRegressor(base_estimator=lasso, n_estimators=10, bootstrap=True, verbose=0, random_state=random_state)

    model.fit(X_train_scaled, y_train)

    feature_names = list(set(X_train_scaled.columns.tolist()))
    counts = dict.fromkeys(feature_names, 0)
    coefs = dict(zip(feature_names, ([] for _ in feature_names)))

    for m in model.estimators_:
        coefficients = m.coef_
        for f, c in zip(feature_names, coefficients[0]):
            coefs[f].append(c)
            if c != 0:
                counts[f] += 1

    means = dict.fromkeys(feature_names, None)
    std_devs = dict.fromkeys(feature_names, None)

    for k, v in coefs.items():
        means[k] = statistics.mean(v)
        std_devs[k] = statistics.stdev(v)

    l95 = []
    u95 = []
    sig = []
    for data in coefs.values():
         conf_it = st.t.interval(alpha=0.95, df=len(data)-1, loc=statistics.mean(data), scale=st.sem(data))

         if conf_it[0] <= 0 and conf_it[1] >= 0:
             sig.append(False)
         elif np.isnan(conf_it[0]) or np.isnan(conf_it[1]):
             sig.append(False)
         else:
             sig.append(True)

         l95.append(conf_it[0])
         u95.append(conf_it[1])

    imp = permutation_importance(model, X_test_scaled, y_test, n_repeats=3, random_state=random_state)

    output = pd.DataFrame()
    output['study_name'] = [study]*len(feature_names)
    output['feature_name'] = feature_names
    output['coef'] = means.values()
    output['coef_sd'] = std_devs.values()
    output['lower_95'] = l95
    output['upper_95'] = u95
    output['count'] = counts.values()
    output['significant'] = sig
    output['permutation_importance'] = imp.importances_mean
    #print(output)

    remove = output[output['significant'] == False]['feature_name']
    #print(remove)

    if len(remove) < len(X_train_scaled.columns):
       LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
       LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]
    else:
       print('LASSO COULDNT DECIDE')
       LASSO_train = X_train_scaled
       LASSO_test = X_test_scaled

    return LASSO_train, LASSO_test, output, model

def run_LASSO3(X_train_scaled, X_test_scaled, y_train, y_test, study=None, param_grid=None, random_state=872510):
    lasso = Lasso(random_state=random_state)
    model = BaggingRegressor(base_estimator=lasso, n_estimators=10, bootstrap=True, verbose=0, random_state=random_state)

    model.fit(X_train_scaled, y_train)

    feature_names = list(set(X_train_scaled.columns.tolist()))
    counts = dict.fromkeys(feature_names, 0)
    coefs = dict(zip(feature_names, ([] for _ in feature_names)))

    for m in model.estimators_:
        coefficients = m.coef_
        for f, c in zip(feature_names, coefficients[0]):
            coefs[f].append(c)
            if c != 0:
                counts[f] += 1

    means = dict.fromkeys(feature_names, None)
    std_devs = dict.fromkeys(feature_names, None)

    for k, v in coefs.items():
        means[k] = statistics.mean(v)
        std_devs[k] = statistics.stdev(v)

    l95 = []
    u95 = []
    sig = []
    for data in coefs.values():
         conf_it = st.t.interval(alpha=0.95, df=len(data)-1, loc=statistics.mean(data), scale=st.sem(data))

         if conf_it[0] <= 0 and conf_it[1] >= 0:
             sig.append(False)
         elif np.isnan(conf_it[0]) or np.isnan(conf_it[1]):
             sig.append(False)
         else:
             sig.append(True)

         l95.append(conf_it[0])
         u95.append(conf_it[1])

    imp = permutation_importance(model, X_test_scaled, y_test, n_repeats=3, random_state=random_state)

    output = pd.DataFrame()
    output['study_name'] = [study]*len(feature_names)
    output['feature_name'] = feature_names
    output['coef'] = means.values()
    output['coef_sd'] = std_devs.values()
    output['lower_95'] = l95
    output['upper_95'] = u95
    output['count'] = counts.values()
    output['significant'] = sig
    output['permutation_importance'] = imp.importances_mean
    #print(output)

    remove = output[output['significant'] == False]['feature_name']
    #print(remove)

    if len(remove) < len(X_train_scaled.columns):
       LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
       LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]
    else:
       print('LASSO COULDNT DECIDE')
       LASSO_train = X_train_scaled
       LASSO_test = X_test_scaled

    return LASSO_train, LASSO_test, output, model


def run_Multinomial1(X_train_scaled, X_test_scaled, y_train, y_test):###872510-1992
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)
    model.fit(X_train_scaled, y_train)
    # define the model evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
    # report the model performance
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
    
    yhat = model.predict_proba(X_test_scaled)
    y_pred = model.predict(X_test_scaled)
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    # Generate detailed classification report
    report =classification_report(y_test, y_pred,output_dict=True)
    print("\nClassification Report:")
    print(report)
    
    
    # Calculate permutation importance
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    abs_importances = np.abs(importances)
    indices = np.argsort(abs_importances)[::-1][:20]
    top_20_features = [X_test_scaled.columns[i] for i in indices]
    top_20_importances = importances[indices]
    # Plot the top 20 features and their importance
    plt.figure(figsize=(10, 6))
    plt.barh(top_20_features, top_20_importances, color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    plt.show()
   

    return model,report, importances,top_20_features






