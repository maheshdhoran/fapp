from flask import Flask,render_template,url_for,request,jsonify
# Random Forest Classification

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():

	

	# Importing the dataset
	dataset = pd.read_csv('data/Social_Network_Ads.csv')
	X = dataset.iloc[:, [2, 3]].values
	y = dataset.iloc[:, 4].values



	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

	# Feature Scaling
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)

	# Fitting Random Forest Classification to the Training set
	from sklearn.ensemble import RandomForestClassifier
	classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
	classifier.fit(X_train, y_train)

	# Predicting the Test set results
	y_pred = classifier.predict(X_test)

	# Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix
	cm = confusion_matrix(y_test, y_pred)

	# opening a pickle file for the classifier
	
	
	classifier2 = pickle.load(open('models/model2.pkl','rb'))


	if request.method == 'POST':
		age = int(request.form['age'])
		salary = int(request.form['salary'])
		x=[[age,salary]]
		result = classifier2.predict(sc.transform(x))
		connect=5
	return render_template('index.html',result = result, connect=connect, age=age, salary=salary)


if __name__ == '__main__':
	app.run(debug=True)
