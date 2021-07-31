import sqlite3
from pathlib import Path
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import sys


# Main Function to run
def main():
    ''' Data Loading '''

    folder = Path("data/score.db")
    con = sqlite3.connect(folder) # connect to the database
    cur = con.cursor() 

    # Shows table names
    cur.execute("SELECT * FROM sqlite_master where type='table'")
    print(cur.fetchall)

    # read the database into a pandas dataframe
    df = pd.read_sql_query('Select * from score',con)

    # Close the connection to the database
    con.close()

    ''' End of Data Loading'''
    ''' Data Cleaning & Feature Engineering'''

    # Calculating hours of sleep, using inbuilt datetime functions
    df['sleep'] = (pd.to_datetime(df['wake_time']) - pd.to_datetime(df['sleep_time']) +timedelta(hours=24))% timedelta(hours=24)
    df['sleep'] = df['sleep'].dt.components['hours']

    # Filter out the students who do not have a final score
    df = df[df['final_test'].notnull()] 

    # Setting any students below 15 to be of age 15
    df.loc[df['age'] < 15, 'age'] = 15

    # Feature Engineering for number of classmates
    df['classmates'] = df['n_male'] + df['n_female']

    # Feature Engineering for type of school
    def schtype(row):
        if row['n_male'] == 0 and row['n_female'] == 0:
            return 'S' # Single student
        elif row['n_male'] == 0:
            return 'F' # Female Only
        elif row['n_female'] == 0:
            return 'M' # Male Only
        else:
            return 'N' # Normal, mixed school

    df['schtype'] = df.apply(schtype,axis =1)

    # Data cleaning for CCA
    df.loc[df['CCA'] == 'Arts', 'CCA'] = 'ARTS'
    df.loc[df['CCA'] == 'Clubs', 'CCA'] = 'CLUBS'
    df.loc[df['CCA'] == 'None', 'CCA'] = 'NONE'
    df.loc[df['CCA'] == 'Sports', 'CCA'] = 'SPORTS'

    # Data cleaning for Tuition
    df.loc[df['tuition'] == 'Yes', 'tuition'] = 'Y'
    df.loc[df['tuition'] == 'No', 'tuition'] = 'N'

    ''' End of Data Cleaning and Feature Engineering'''
    
    '''Defining reusable functions & variables for both Approaches'''

    # Declaring categorical and numerical features, 12 in total.
    cat = ['direct_admission','CCA','learning_style','gender','tuition','schtype'] # 6 total categorical features
    num = ['number_of_siblings','age','hours_per_week','attendance_rate','sleep','classmates'] # 6 total numerical features

    # Function for preprocessing data in training
    def preprocess(X,y,flag): 
        scaler = StandardScaler()
        target_scaler = MinMaxScaler()
        
        '''One hot encoding categorical variables'''
        X = pd.get_dummies(X,['direct_admission','CCA','learning_style','gender','tuition','schtype'])
        
        
        ''' Split dataset into test and training'''
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42) # For reproducibility
        
        '''Impute missing values in attendance_rate, without any spillover from training data'''
        X_train['attendance_rate'].fillna(X_train['attendance_rate'].median(), inplace = True)
        X_test['attendance_rate'].fillna(X_test['attendance_rate'].median(), inplace = True)
        
        '''Scaling of numerical data'''
        X_train[num] = scaler.fit_transform(X_train[num])
        X_test[num] = scaler.transform(X_test[num])
        
        if flag == 'reg': #If the problem is regression, the target needs to be scaled as well
            y_train = target_scaler.fit_transform(y_train.values.reshape(-1,1))
            y_test = target_scaler.transform(y_test.values.reshape(-1,1))
            return X_train, X_test, y_train, y_test, target_scaler 
        
        return X_train, X_test, y_train, y_test

    # Function for preprocessing entire dataframe
    def preprocesstable(table):
        scaler = StandardScaler()
        
        '''One hot encoding categorical variables'''
        table = pd.get_dummies(table,['direct_admission','CCA','learning_style','gender','tuition','schtype'])
        
        '''Impute missing values in attendance_rate, without any spillover from training data'''
        table['attendance_rate'].fillna(table['attendance_rate'].median(), inplace = True)
        
        '''Scaling of numerical data'''
        table[num] = scaler.fit_transform(table[num])
        
        return table

    '''End of Reusable Functions & Variables'''

    ''' Start of Approach 1 (Regression)'''
    if (int(sys.argv[1]) == 1):

        # Importing Required Functions
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor
        from sklearn import svm
        from sklearn.linear_model import SGDRegressor
        from sklearn import linear_model
        from sklearn.neighbors import KNeighborsRegressor

        # Extracting the features that will be used for training
        table = df[['number_of_siblings','direct_admission','CCA','learning_style','gender','tuition','age','hours_per_week',
                'attendance_rate','sleep','classmates','schtype']]

        # Target is the variable we want to predict, the student's test scores
        target = df['final_test']


        # Defining functions for 4 different ML algorithms
        def Random_Forest(X_train, X_test, y_train, y_test):
            RFF = RandomForestRegressor(random_state = 42)
            RFF.fit(X_train, y_train.ravel())
            RFF_predictions = RFF.predict(X_test)
            print('Random Forest Regressor score is %f' % RFF.score(X_test, y_test))
            return RFF
            
        def Support_Vector(X_train, X_test, y_train, y_test):
            SVMreg = svm.SVR()
            SVMreg.fit(X_train, y_train.ravel())
            predictions =SVMreg.predict(X_test)
            print('Support Vector Regressor score is %f' % SVMreg.score(X_test, y_test))
            return SVMreg

        def linear_regression(X_train, X_test, y_train, y_test):
            lm = linear_model.LinearRegression()
            lm.fit(X_train, y_train)
            lm_predictions = lm.predict(X_test)
            print('Linear Regression score is %f' % lm.score(X_test, y_test))
            return lm
            
        def KNearest_neighbor(X_train, X_test, y_train, y_test):
            KNNR = KNeighborsRegressor()
            KNNR.fit(X_train,y_train)
            print('K-Nearest Neighbors Regressor score is %f' % KNNR.score(X_test, y_test))
            return KNNR

        # Preprocessing and assigning into train and test sets
        X_train, X_test, y_train, y_test, scaler = preprocess(table,target,'reg')

        '''Model 1 Results'''
        print('------ Approach 1 Scores -----')

        RFF = Random_Forest(X_train, X_test, y_train, y_test)
        SVMreg = Support_Vector(X_train, X_test, y_train, y_test)
        lm = linear_regression(X_train, X_test, y_train, y_test)
        KNNR = KNearest_neighbor(X_train, X_test, y_train, y_test)

        '''RFR results analysis'''
        # Preprocess the entire dataframe
        table = preprocesstable(table) 

        # Regression prediction for the whole dataframe
        RFF_predictions = RFF.predict(table)


        df['predicted_test'] = scaler.inverse_transform(RFF_predictions.reshape(-1,1))
        print('\n ----- Results Analysis for Approach 1 using Random Forest Regression -----')
        print('Mean Absolute Error for Approach 1: ' 
            +"{:.2f}".format(mean_absolute_error(df['final_test'],df['predicted_test'])))
        print('Mean Squared Error for Approach 1: ' 
            + "{:.2f}".format(mean_squared_error(df['final_test'],df['predicted_test'])))

    ''' End of Approach 1 (Regression)'''



    ''' Start of Approach 2 (Classification)'''
    if (int(sys.argv[1]) == 2):
        
        # Importing Required Functions
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, make_scorer
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier

        # Applying a grade to every student in the dataset, based on their score
        def grading(row):
            score = row['final_test']
            if score >= 74:
                return 'Good' # High Distinction
            elif score >= 60:
                return 'Average' # Distinction
            else: 
                return 'Needs Help' # Merit

        df['grade'] = df.apply(grading,axis =1)

        # Features are unchanged from before
        table = df[['number_of_siblings','direct_admission','CCA','learning_style','gender','tuition','age','hours_per_week',
                'attendance_rate','sleep','classmates','schtype']]

        # Target variable is now the grade column
        target = df['grade']

        # Preprocessing the data, using flag as classification
        X_train, X_test, y_train, y_test = preprocess(table,target,'class')

        # Define a function to plot confusion matrix and calculate average accuracy and f1_score
        def plotcm(model, predictions,X_test, y_test): # Function to help plot confusion matrix, calculate score and f1 score
            cm = confusion_matrix(y_test,predictions,labels = model.classes_)
            disp = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
            print('------ ' + str(model) + ' ------')
            disp.plot()
            print('Average Accuracy: ',"{:.4f}".format(model.score(X_test,y_test)))
            print('Average f1_score: ',
                "{:.4f}".format(f1_score(y_test,predictions,labels = model.classes_, average='weighted')))

        # Simple decision tree classifier
        dtree_model = DecisionTreeClassifier().fit(X_train, y_train)
        dtree_predictions = dtree_model.predict(X_test)
        
        plotcm(dtree_model, dtree_predictions,X_test,y_test)

        # Support Vector Machine Classifier
        svm_model = SVC().fit(X_train, y_train)
        svm_predictions = svm_model.predict(X_test)

        plotcm(svm_model, svm_predictions,X_test,y_test)
        
        # Random Forest Classifier (Which has been pre-tuned)
        RFC_model = RandomForestClassifier(criterion= 'entropy',max_depth= 70, min_samples_split= 10, n_estimators= 2500,random_state=42).fit(X_train,y_train)
        RFC_predictions = RFC_model.predict(X_test)

        plotcm(RFC_model, RFC_predictions,X_test,y_test)

        '''Analysis of Classification Model'''
        # Preprocess the entire dataframe
        table = preprocesstable(table) 
        df['predicted_grade'] = RFC_model.predict(table)

        # Finding all the wrongly classified students
        print('Total wrongly classified students: ' + str(len(df[df['grade'] != df['predicted_grade']])))
        
        # Percentage of wrongly classified
        print('Total students: '+ str(len(df)))
        print('Percentage of wrongly classified students: ' + 
            "{:.2f}".format(len(df[df['grade'] != df['predicted_grade']])/len(df) *100) +'%')
        
        # Occurrences of Good students predicted to require help
        print('Students who performed well but were predicted to underperform: ' +
            str(len(df[(df['grade'] == 'Good') & (df['predicted_grade']=='Needs Help')]))) 
        
        # Occurrences of students who required help, but actually performed well
        print('Students who underperformed but were predicted to do well: ' +
            str(len(df[(df['predicted_grade'] == 'Good') & (df['grade']=='Needs Help')]))) 
        
    '''End of Approach 2 (Classification)'''

# Driver code to execute the program   
if __name__ == '__main__':
	main()
