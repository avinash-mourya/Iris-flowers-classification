from flask import Flask,render_template,request
import numpy as np
import pickle
app = Flask(__name__)
print(__name__)
 #######-----------------Model----------------------       
@app.route('/',methods=['GET','POST'])
def model():
     if(request.method=="POST"):
        petal_width = request.form.get('petal_width')
        petal_length = request.form.get('petal_length')
        sepal_width = request.form.get('sepal_width')
        sepal_length = request.form.get('sepal_length')
        iris_list = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
        classifier = pickle.load(open('xgb_iris.pkl','rb'))
        pred = classifier.predict(iris_list)
        return render_template('iris_pred.html',prediction = pred[0],req=(request.method=="POST"))
     else:
         return render_template('iris_pred.html',req =(request.method=="POST"))
         
if __name__ == "__main__":
     app.run()
   
