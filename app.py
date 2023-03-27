from flask import Flask, request, jsonify
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
import pickle 


# Create a SparkSession
spark = SparkSession.builder.appName('LoanPrediction').getOrCreate()

#  load the pre-trained model.
with open(f'model.pkl', 'rb') as f:
    model = pickle.load(f)

# create a Flask application
app = Flask(__name__)


# create a Flask route for loan prediction
@app.route("/loan_prediction", methods=["POST"])
def predict_loan():
    # parse the loan application data from the request object

        Gender= request.form['Gender']
        Married = request.form['Married']
        Dependents = request.form['Dependents']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        ApplicantIncome = request.form['ApplicantIncome']
        CoapplicantIncome = request.form['CoapplicantIncome']
        LoanAmount = request.form['LoanAmount']
        Loan_Amount_Term = request.form['Loan_Amount_Term']
        Credit_History = request.form['Credit_History']
        Property_Area = request.form['Property_Area']
        
       

        df = spark.createDataFrame([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount,
                                             Loan_Amount_Term, Credit_History, Property_Area]],columns=['Gender', 'Married', 'Dependents', 'Education',
                                                                                                        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
                                                                                                        'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
                                                                                                        'Property_Area'])
        # Select the string columns
        string_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

        # Convert the string columns to numeric columns using StringIndexer
        for col_name in string_cols:
            indexer = StringIndexer(inputCol=col_name, outputCol=col_name+'_num')
            df = indexer.fit(df).transform(df)
            df = df.drop(col_name)

        # Select the numeric columns you want to scale
        numeric_cols = ['Gender_num', 'Married_num', 'Dependents', 'Education_num','Self_Employed_num','ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area_num']

        # Convert integer features to double
        for col_name in numeric_cols:
            df = df.withColumn(col_name, col(col_name).cast('double'))

        # Create a feature vector using VectorAssembler
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol='features')
        df = assembler.transform(df)

        # Scale the numeric features
        scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)

        # make a prediction
        prediction = model.predict(df)

        # return the prediction as a response
        return jsonify({"prediction": prediction})


       
        if prediction == 0:
            result = 'NO'
        else:
            result = 'YES'

        return render_template('main.html',result=prediction)


if __name__ == '__main__':
    app.run()
