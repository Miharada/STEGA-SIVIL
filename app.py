from flask import Flask, render_template, request, flash, redirect, session, g, url_for
from io import BytesIO
from werkzeug.utils import secure_filename

import numpy as np

import cv2

import sqlite3
import pandas as pd

from extractVal import RealValidation
from embed import Embedding

import sqlite3

app = Flask(__name__)
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['png'])


class User:
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password
    def __repr__(self):
        return f'<User: {self.username}>'

users = []
users.append(User(id = 1, username="superadmin", password="123456"))

@app.before_request
def before_request():
    g.user = None
    if 'user_id' in session:
        user = [x for x in users if x.id == session['user_id']][0]
        g.user = user

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# def createDB(dbname):
#   db = sqlite3.connect(dbname+".db")
#   db.execute("drop table if exists validation")
#   try:
#     db.execute("create table validation(id,seed)")
#   except:
#     print("Table Already Exists !!!")

# db = createDB("val")


@app.route('/')
def home_page():
    return render_template('validationPage.html')
    
@app.route('/', methods=['POST'])
def uploadDiplomas():
    if 'file' not in request.files:
        flash("No file part")
    file = request.files['diploma']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        #https://stackoverflow.com/questions/27517688/can-an-uploaded-image-be-loaded-directly-by-cv2
        img = cv2.imdecode(np.frombuffer(request.files['diploma'].read(), np.uint8), cv2.IMREAD_UNCHANGED)

        if (RealValidation(img)):
            valid = True
            invalid = False
        else:
            valid = False
            invalid = True
    else:
        flash("Allowed image types is .png")
        return redirect(request.url)
    # print(hasil)
    return render_template('validationPage.html', show_predictions_modal=True, valid=valid, invalid=invalid)

@app.route('/adminpage')
def adminPages():
    if not g.user:
        return redirect(url_for('loginAdmin'))
    return render_template("adminPageEmbed.html")

@app.route('/adminpage', methods=['POST'])
def embedDiplomas():
    if 'file' not in request.files:
        flash("No file part")
    file = request.files['diploma']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        number = request.form.get('dipNum', False)
        ids = request.form.get('dipID', False)
        img = cv2.imdecode(np.frombuffer(request.files['diploma'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = np.array(img)[:,:,:-1]
        emb_img = Embedding(img, ids, number)
        cv2.imwrite("Embedded Diploma/Diploma_"+ids+".png", emb_img)
    return render_template("adminPageEmbed.html")

@app.route('/dbpage')
def dbpages():
    if not g.user:
        return redirect(url_for('loginAdmin'))
    data = getDb()
    # print(data)
    return render_template("adminPageDb.html", datas=data)

@app.route('/loginadmin', methods=['GET','POST'])
def loginAdmin():
    if request.method == 'POST':
        session.pop('user_id', None)
        username = request.form['Username']
        password = request.form['Password']

        user = [x for x in users if x.username == username][0]
        if user and user.password == password:
            session['user_id'] = user.id
            return redirect(url_for('adminPages'))

        return redirect(url_for('loginAdmin'))

    return render_template("loginAdminpages.html")



def getDb():
    db = sqlite3.connect('val.db')
    return pd.read_sql_query("SELECT * FROM validation", db)
    
@app.route('/dbpage', methods=['GET','POST'])
def removeData():
    if request.method == 'POST':
        myid = request.form['idtodel']
        db = sqlite3.connect("val.db")
        cursor = db.cursor()
        query = "DELETE FROM validation WHERE id='{}'".format(myid)
        cursor.execute(query)
        db.commit()
        cursor.close()
        return redirect(url_for('dbpages'))


if __name__ == "__main__":
    app.run(debug=False)



#https://roytuts.com/upload-and-display-image-using-python-flask/
#https://medium.com/@marioruizgonzalez.mx/how-install-tesseract-orc-and-pytesseract-on-windows-68f011ad8b9b

