from flask import Flask, render_template, request, send_from_directory
from main import *

app = Flask(__name__)

df = read_data()
X_train, X_test, y_train, y_test = data_preprocessing(df)
clff = None

@app.route('/')
def ss():
    return render_template('login.html')


@app.route('/train',methods = ['POST', 'GET'])
def train():
   if request.method == 'POST':
      
      algo_num = request.form['classifier']
      global clff

      if algo_num == '1':
         clff = logestic_training(X_train,y_train)
      elif algo_num == '2':
         clff = knn_training(X_train,y_train)
      elif algo_num == '3':
         clff = tree_training(X_train,y_train)
      elif algo_num == '4':
         clff = forest_training(X_train,y_train)         
      else :
         return 'method has not been trained'

      return 'method has been trained '
   

@app.route('/model_evaluation',methods = ['POST', 'GET'])
def evaluat():

   res = test_classifier (clff,X_test , y_test)
   N_df = algo_eval(clff ,X_test , y_test).split('\n')
   return render_template('evaluation.html',name=res,l1=N_df[0],l2=N_df[2],l3=N_df[3],l4=N_df[5])  
   

@app.route('/test_row',methods = ['POST', 'GET'])
def test_row():
   if request.method == 'POST':
      l_var = []
      l_var.append(request.form['sen']) 
      l_var.append(request.form['part'])
      l_var.append(request.form['dep'])
      l_var.append(request.form['tenure'])
      l_var.append(request.form['phon'])
      l_var.append(request.form['mult'])
      l_var.append(request.form['intr'])
      l_var.append(request.form['onl'])
      l_var = list(map(int, l_var))
      l_var = np.array(l_var).reshape(1, -1)
      d=row_probability(clff,l_var)
      
      return render_template('test.html',name=d)  



if __name__ == '__main__':
   app.run(debug = True)



















