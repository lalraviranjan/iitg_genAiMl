# %%
from flask import Flask, render_template, redirect, url_for, request, session
import requests
import os

# %%
app = Flask(__name__)
# app.secret_key = os.urandom(24)

if __name__ == '__main__':
    app.run(debug=True, port=8080)

# %%
@app.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        response = requests.post('http://localhost:8000/api/login', json={"username": username, "password": password})
        
        if response.status_code == 200:
            session['user_id'] = response.json()['user_id']
            return redirect(url_for('main'))
        else:
            return "Invalid credentials", 401
    return render_template('login.html')


