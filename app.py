from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
app.config['UPLOAD_FOLDER'] = 'uploads/'
df,X,y = None,None,None
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def dimension(df):
    rows = len(df.axes[0])
    cols = len(df.axes[1])
    return rows, cols

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if username == 'user' and password == 'pass':
        session['user'] = username
        return redirect(url_for('buttons'))
    else:
        flash('Login failed. Incorrect username or password.', 'error')
        return redirect(url_for('home'))

@app.route('/index')
def buttons():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/ingest', methods=['POST'])
def ingest():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        session['uploaded_file'] = file.filename  # Store the filename in session
        flash(f'File {file.filename} uploaded successfully.')

        try:
            global df
            df = pd.read_csv(file_path)
            rows, cols = dimension(df)
            first_rows = df.head(5).to_html()
            return render_template('index.html', filename=os.path.basename(file_path), rows=rows, cols=cols, first_rows=first_rows, features=df.columns.tolist())
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)
    
    return render_template('index.html')

@app.route('/remove_features', methods=['POST'])
def remove_features():
    global df
    try:
        data = request.get_json()
        features_to_remove = data.get('features', [])
        print(features_to_remove)
        
        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        # Ensure features_to_remove is a list
        if not isinstance(features_to_remove, list):
            return jsonify({'error': 'Invalid input format'}), 400

        # Remove the selected features from the DataFrame
        cleaned_features_to_remove = [feature.strip('×').strip() for feature in features_to_remove]
        removed_features = [feature for feature in cleaned_features_to_remove if feature in df.columns]
        df.drop(columns=removed_features, inplace=True)
        
        # Prepare response data
        response_data = {
            'success': True,
            'rows': df.shape[0],
            'cols': df.shape[1],
            'features': list(df.columns),
            'first_rows': df.head().to_html(classes='table table-striped')
        }
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/convert_to_numbers', methods=['POST'])
def convert_to_numbers():
    global df
    try:
        data = request.json
        feature = data.get('feature')
        print(f"Received feature: {feature}")  # Debugging statement

        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        if feature in df.columns:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])
            
            # Generate the HTML for the first 5 rows of the updated DataFrame
            first_rows = df.head(5).to_html(classes='dataframe', border=0, index=False)
            
            return jsonify({
                'success': True,
                'message': 'Feature converted successfully!',
                'rows': df.shape[0],
                'cols': df.shape[1],
                'first_rows': first_rows,
                'features': list(df.columns)  # Include the updated list of features
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Feature not found in DataFrame.'
            })
    except Exception as e:
        # Log the exception for debugging purposes
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/select_label', methods=['POST'])
def select_label():
    global df, X, y
    try:
        data = request.json
        label = data.get('label')
        print(f"Selected label: {label}")  # Debugging statement

        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        if label in df.columns:
            y = df[label]
            X = df.drop(columns=[label])
            print(X,y)
            
            # Generate the HTML for the first 5 rows of the DataFrame
            first_rows = df.head(5).to_html(classes='dataframe', border=0, index=False)

            return jsonify({
                'success': True,
                'label': label,
                'first_rows': first_rows
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Label not found in DataFrame.'
            })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train_test_split', methods=['POST'])
def train_test_split_route():
    print('hi')
    try:
        data = request.json
        train_data = float(data['train_data'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_data / 100)
        app.config['X_train'] = X_train
        app.config['X_test'] = X_test
        app.config['y_train'] = y_train
        app.config['y_test'] = y_test
        print('accuracy',train_data)
        return jsonify({'accuracy': 'Train-Test Split Successful'})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/train_decision_tree', methods=['POST'])
def train_decision_tree():
    try:
        data = request.json
        
        criterion = data['criterion']
        max_depth = int(data['max_depth'])
        min_sample_split = int(data['min_sample_split'])
        
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_sample_split)
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        print("Desicion Tree")
        accuracy = accuracy_score(app.config['y_test'], y_pred)
        print(accuracy)
        return jsonify({'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_ada_boost', methods=['POST'])
def train_ada_boost():
    try:
        data = request.json
        n_estimators = int(data['n_estimators'])
        
        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)
        return jsonify({'accuracy': accuracy})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_svm', methods=['POST'])
def train_svm():
    data = request.json
    kernel = data['kernel']
    c_param = float(data['c_param'])
    gamma = data['gamma']

    model = SVC(kernel=kernel, C=c_param, gamma=gamma)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify(message=f"SVM accuracy: {accuracy:.2f}")

@app.route('/train_bagging', methods=['POST'])
def train_bagging():
    data = request.json
    n_estimators = int(data['n_estimators'])
    max_samples = float(data['max_samples'])
    max_features = float(data['max_features'])

    model = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify(message=f"Bagging accuracy: {accuracy:.2f}")

@app.route('/train_random_forest', methods=['POST'])
def train_random_forest():
    data = request.json
    rf_criterion = data['rf_criterion']
    rf_estimators = int(data['rf_estimators'])
    rf_max_depth = int(data['rf_max_depth'])
    rf_max_features = data['rf_max_features']

    model = RandomForestClassifier(criterion=rf_criterion, n_estimators=rf_estimators, max_depth=rf_max_depth, max_features=rf_max_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify(message=f"Random Forest accuracy: {accuracy:.2f}")

@app.route('/train_xgboost', methods=['POST'])
def train_xgboost():
    data = request.json
    num_boost_round = int(data['num_boost_round'])
    early_stopping = int(data['early_stopping'])

    model = XGBClassifier(n_estimators=num_boost_round)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train, early_stopping_rounds=early_stopping, eval_set=[(X_test, y_test)], verbose=False)
    accuracy = model.score(X_test, y_test)
    return jsonify(message=f"XGBoost accuracy: {accuracy:.2f}")

@app.route('/train_stacking', methods=['POST'])
def train_stacking():
    data = request.json
    stacking_cv = int(data['stacking_cv'])
    final_estimator = data['final_estimator']

    base_models = [
        ('dt', DecisionTreeClassifier()),
        ('svm', SVC(probability=True))
    ]
    model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return jsonify(message=f"Stacking accuracy: {accuracy:.2f}")

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('uploaded_file', None)  # Remove uploaded file info from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
