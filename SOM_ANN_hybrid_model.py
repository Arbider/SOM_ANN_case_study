# Experimentation and Optimization of a Hybrid Deep Learning Model (SOM/ANN)

"""
DISCLAIMER:
The original concept of this project originates from the Deep Learning A-Z™ 2023: Neural Networks, AI & ChatGPT Bonus
course given by Kirill Eremenko and Hadelin de Ponteves on Udemy. I wish to make clear that the original assignment and
the knowledge I've obtained to complete this program has come from both the teachings offered by this course and my own 
personal research and experimentation, and this program is merely intended as a representation of such knowledge to whom 
it may concern. I give credit where it is due for the experience I've gained from those courses and the ressources they've 
provided students like me.
"""

# Part 1 - Identifying the potential fraudulent customers of a hypothetical bank with a Self-Organizing Map
###############################################################################################################################

# Importing the libraries
import numpy as np  
import pandas as pd 

# Importing the dataset (* This dataset is course-provided material and represents customers to a bank that have or have not been approved for a credit card)
dataset = pd.read_csv('Credit_Card_Applications.csv') # The splitting of the data into X and y has nothing to do with supervised learning and testing the accuracy of the model, but rather we will just use the values stored in X and y to distinguish the approved/non-approved customers.
X = dataset.iloc[:, :-1].values # We take all the rows, and all the columns except the last one.
y = dataset.iloc[:, -1].values # And here we take all the rows, but only take the last column representing the approval or not of customers (Class column).

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X) # Now X is fitted (it is normalized between 0 and 1)

# Training the SOM
from minisom import MiniSom # This is an SOM library that is built for and makes use of NumPy matrices ! Citation at the bottom of the page.
# Below, we leave sigma (the radius of the different neighbourhoods in the grid), the learning rate, the decay function, and the random seed at their default value.
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5) # The input length is the number of columns used as input from the columns in X.
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show # Here, pylab is used for the visualization of the results, but I wish to upgrade asap.
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's'] # o will generate a circle shape, and s will generate a square. 
colors = ['r', 'g'] # r will generate the colour red, and g will generate the colour green. Red circles stand for non-approvals and green squares stand for approvals.
# Below, the i variable will take on the values of the indexes of the customers and x will take on all the vectors of the customers' info at each iteration.
for i, x in enumerate(X):
    w = som.winner(x) # Here, 'winner' is the computer for the coordinates of the winning neurons that stand out from the samples from x.   
    # Here we determine which shape (o or s) and colour (r or g) the customer has based on the index of customer y[i]. The y values represent a Boolean value (in the Class column) of either 0 or 1 that indicates credit card approval or not, thus you obtain that value associated with the customer at index i in y. Finally, the markers/color list element is chosen accordingly as a red circle for rejected cases and green squares for accepted ones.
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]], 
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2) 
    
show()

"""
Run all the above lines up to this point in the interactive 
window if using an IDE like VSCode to determine which coordinates 
you wish to input as the mappings in the following lines of code. 
Since the SOM makes different groupings every time the program is run, 
the above process has to also be repeated every time.
"""

# Finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate(((mappings[(2,6)]), (mappings[(3,9)]), (mappings[(2,9)]), (mappings[(7,9)])), axis = 0) # Always run all the lines before this one and take note of the mappings of interest since you might end up with only a single set of mappings or an empty array that don't require concatenation.
# frauds = mappings[(6,9)] # Depending on your personal parameters, you can always choose to have minimum 2 sets of coordinates based off what is observed on the map. I you choose to sample less, you don't have to concatenate anything here, but you are more likely to end up with weaker final results in the ANN section.
frauds = sc.inverse_transform(frauds) 

# Print the fraud customer IDs
print('Fraud Customer IDs')
for i in frauds[:, 0]:
    print(int(i))



# Part 2 - Going from Unsupervised to Supervised Deep Learning
###############################################################################################################################

# Implementation of the Atrificial Neural Network
"""
Process:
    Since we've determined the 'suspicious' customers as likely frauds, use their IDs
    as true values for the ANN to be trained on. (i.e. show it that customers with similar 
    and respective tendencies are likely frauds, and so it should predict potential future frauds.).
    For testing and trial purposes, you could also simply assume the sample of frauds determined from the SOM
    since the data above comes from a K-means clustering algorithm and in practice is as hypothetical as 
    using random customers as samples for the ANN below
"""

# Importing TensorFlow
import tensorflow as tf 

# Data Preprocessing
# Reusing the dataset to create the matrix containing the customer's features.
customers = dataset.iloc[:, 1:].values

# Using the list of potential fraudulent customers'Customer IDs to create an array for the dependant variables in is_fraud. This array will be set as 0s and turned into 1s when the fraudulent Customer IDs are also matched with the corresponding ones in the dataset. This keeps the is_fraud array without need of scaling/fitting.  
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Building the ANN
# Initialize the ANN, add input layer, first and second hidden layer, and output layer. After trial and error, I found that the following structure performs adequately.
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
ann.add(tf.keras.layers.Dense(units=9, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set.
ann.fit(customers, is_fraud, batch_size = 1, epochs = 10) # This is where the dependent variable is used to train the model on its relations with the gross data. On this small of a dataset, you would not require mmany more epochs than here for accurate results.



# Expected Result: Ranking of the predicted probabilities that each customer committed fraud.
###############################################################################################################################

# Making the predictions and evaluating the model 
y_pred = ann.predict(customers) # Now the ANN is making decisions based off the training as to how much each person is likely to be comitting fraud.
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis = 1) # The following will provide a ranking for the most likely fraudulent customers to the least likely.
y_pred = y_pred[y_pred[:, 1].argsort()] # Here the columns are both sorted in according to the ascending order of column at index 1.
# For more human readable results, multiply the entire column[1] results in y_pred by 100.
print('These are the percentages of probabilities in ascending order of each customer\'s likelihood of them actively committing fraud:')
for i in range(len(y_pred)):
    y_pred[i, 1] = format(y_pred[i, 1] * 100, '.2f')
    print(y_pred[i, 1], '%') 

""" 
Make a confusion matrix comparing the predicted fraudulent customers and the Class Column from the dataset containing 1s and 0s, 
1 being a customer approved for a credit card and 0 being non-approved customers. With the matrix, we can observe the hypothetical
current effectiveness of the bank at processing their customers. Here we'll compare the results observed as distant from the norm 
in the SOM stored in the is_fraud variable and the Class column from the dataset now stored in the approvals variable.
"""

# Making the confusion matrix.
from sklearn.metrics import confusion_matrix, accuracy_score
approvals = dataset.iloc[:, -1].values
cm = confusion_matrix(approvals, is_fraud)
print(cm)
acc_score = accuracy_score(approvals, is_fraud)
print(format((acc_score * 100), '.0f'), '% of credit card approvals have been granted to customers likely committing fraud.', sep='')



# Citations:
###############################################################################################################################
# @misc{vettigliminisom,
#   title={MiniSom: minimalistic and NumPy-based implementation of the Self Organizing Map},
#   author={Giuseppe Vettigli},
#   year={2018},
#   url={https://github.com/JustGlowing/minisom/},
# }
