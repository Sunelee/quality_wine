
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Title and image
st.title('Wine Quality Prediction')
st.image('wine_quality.png', caption="'Life is too short to drink bad wine' - Anonymous")
st.subheader('**Quality Index:**')
st.markdown('* **1--> 5 = Bad**')
st.markdown('* **6--> 10 = Good**')

# Function to load data from CSV
def get_data(filename):
    data = pd.read_csv(filename, delimiter=';')
    return data

filename = 'winequality-white.csv'
data = get_data(filename)

# Function to preprocess data and create X, y
def get_dataset(data):
    bins = (1, 5, 10)
    groups = ['1', '2']
    data['quality'] = pd.cut(data['quality'], bins=bins, labels=groups)
    x = data.drop(columns=['quality'])
    y = data['quality']
    return x, y

x, y = get_dataset(data)

# Display dataset summary statistics
st.write("### Dataset Summary Statistics")
st.write(data.describe())

# Data visualization expander
with st.expander('Data Visualisation'):
    plot = st.selectbox('Select Plot type', ('Histogram', 'Box Plot', 'Heat Map'))

    if plot == 'Heat Map':
        # Heatmap
        fig1 = plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(data.corr()[['quality']].sort_values(by='quality', ascending=False), vmin=-1,
                              vmax=1, annot=True)
        heatmap.set_title('Features Correlating with quality', fontdict={'fontsize': 18}, pad=16)
        st.pyplot(fig1)
    else:
        # Histogram or Box Plot
        feature = st.selectbox('Select Feature', ('fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                                  'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                                                  'pH', 'sulphates', 'alcohol'))
        if plot == 'Histogram':
            fig2 = plt.figure(figsize=(7, 5))
            plt.xlabel(feature)
            sns.distplot(data[feature])
            st.pyplot(fig2)
        else:
            fig3 = plt.figure(figsize=(3, 3))
            plt.xlabel(feature)
            plt.boxplot(x=data[feature])
            st.pyplot(fig3)

# Prediction expander
with st.expander('Prediction'):
    # Select classifier
    classifier = st.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest', 'XGBoost'))

    # Function to get algorithm-specific parameters
    def get_algorithm(classifier):
        params = dict()
        if classifier == 'KNN':
            n_neighbors = st.slider('n_neighbors', 1, 20)
            params['n_neighbors'] = n_neighbors
        elif classifier == 'SVM':
            C = st.slider('C', 1.0, 15.0)
            params['C'] = C
            kernel = st.selectbox('Select the kernel type', ('linear', 'poly', 'rbf', 'sigmoid'))
            params['kernel'] = kernel
        elif classifier == 'Random Forest':
            n_estimators = st.slider('n_estimators', 100, 1000)
            max_depth = st.slider('max_depth', 1, 15)
            params['n_estimators'] = n_estimators
            params['max_depth'] = max_depth
        else:
            learning_rate = st.slider('learning_rate', 0.001, 0.5)
            max_depth = st.slider('max_depth', 1, 15)
            params['learning_rate'] = learning_rate
            params['max_depth'] = max_depth
        return params

    params = get_algorithm(classifier)

    # Function to update the model based on the selected classifier and parameters
    def model_update(classifier, params):
        if classifier == 'KNN':
            model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
        elif classifier == 'SVM':
            model = SVC(C=params['C'], kernel=params['kernel'])
        elif classifier == 'Random Forest':
            model = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'])
        else:
            model = XGBClassifier(learning_rate=params['learning_rate'], max_depth=params['max_depth'])
        return model

    model = model_update(classifier, params)

    # Standardize features
    sc = StandardScaler()
    x = sc.fit_transform(x)

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # Fit the model
    model.fit(x_train, y_train)

    # Make predictions
    y_predict = model.predict(x_test)
    accuracy1 = accuracy_score(y_test, y_predict)
    accuracy1 = accuracy1 * 100
    accuracy1 = round(accuracy1, 2)
    st.write(f'Accuracy is {accuracy1}%')

# Sidebar for model evaluation metrics
with st.sidebar:
    st.title('Model Evaluation')
    metric = st.selectbox('Select Metric', ('Accuracy', 'Precision', 'F1 Score'))

    if metric == 'Accuracy':
        st.write(f'Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%')
    elif metric == 'Precision':
        st.write(f'Precision: {precision_score(y_test, y_predict, average="weighted") * 100:.2f}%')
    elif metric == 'F1 Score':
        st.write(f'F1 Score: {f1_score(y_test, y_predict, average="weighted") * 100:.2f}%')
