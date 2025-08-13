from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key
app.config['UPLOAD_FOLDER'] = 'uploads/'
df, X, y = None, None, None

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


def dimension(df):
    """Return the dimensions of a DataFrame"""
    return df.shape[0], df.shape[1]


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
            first_rows = df.head(5).to_html(classes='table table-striped', index=False)
            return render_template('index.html',
                                   filename=os.path.basename(file_path),
                                   rows=rows,
                                   cols=cols,
                                   first_rows=first_rows,
                                   features=df.columns.tolist())
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(request.url)

    return redirect(url_for('buttons'))


@app.route('/remove_features', methods=['POST'])
def remove_features():
    global df
    try:
        data = request.get_json()
        features_to_remove = data.get('features', [])

        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        # Remove the selected features from the DataFrame
        cleaned_features_to_remove = [feature.strip('×').strip() for feature in features_to_remove]
        removed_features = [feature for feature in cleaned_features_to_remove if feature in df.columns]
        df.drop(columns=removed_features, inplace=True, errors='ignore')

        # Prepare response data
        response_data = {
            'success': True,
            'rows': df.shape[0],
            'cols': df.shape[1],
            'features': list(df.columns),
            'first_rows': df.head().to_html(classes='table table-striped', index=False)
        }
        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/convert_to_numbers', methods=['POST'])
def convert_to_numbers():
    global df
    try:
        data = request.get_json()
        feature = data.get('feature')

        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        if feature in df.columns:
            le = LabelEncoder()
            df[feature] = le.fit_transform(df[feature])

            return jsonify({
                'success': True,
                'message': 'Feature converted successfully!',
                'rows': df.shape[0],
                'cols': df.shape[1],
                'first_rows': df.head(5).to_html(classes='table table-striped', index=False),
                'features': list(df.columns)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Feature not found in DataFrame.'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/select_label', methods=['POST'])
def select_label():
    global df, X, y
    try:
        data = request.get_json()
        label = data.get('label')

        if df is None:
            return jsonify({'error': 'No DataFrame loaded'}), 400

        if label in df.columns:
            y = df[label]
            X = df.drop(columns=[label])

            return jsonify({
                'success': True,
                'label': label,
                'first_rows': df.head(5).to_html(classes='table table-striped', index=False)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Label not found in DataFrame.'
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


from flask import Flask, session, jsonify, request
import pandas as pd
import pickle
import base64
from io import BytesIO

# Add these global variables at the top
train_test_data = {
    'X_train': None,
    'X_test': None,
    'y_train': None,
    'y_test': None
}


@app.route('/train_test_split', methods=['POST'])
def train_test_split_route():
    global X, y, train_test_data

    try:
        data = request.get_json()
        train_size = float(data.get('train_data', 80)) / 100

        if X is None or y is None:
            return jsonify({'error': 'Please select a label first'}), 400

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            train_size=train_size,
            random_state=42
        )

        # Store in memory instead of session
        train_test_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        return jsonify({
            'success': True,
            'message': f'Split successful (Train: {train_size * 100:.0f}%, Test: {(1 - train_size) * 100:.0f}%)'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_decision_tree', methods=['POST'])
def train_decision_tree():
    global train_test_data

    try:
        # Check if split was performed
        if train_test_data['X_train'] is None:
            return jsonify({'error': 'Please perform train-test split first'}), 400

        data = request.get_json()

        # Get parameters with defaults
        criterion = data.get('criterion', 'gini')
        max_depth = int(data.get('max_depth', 5)) if data.get('max_depth') else None
        min_sample_split = int(data.get('min_sample_split', 2))

        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_sample_split
        )

        model.fit(train_test_data['X_train'], train_test_data['y_train'])
        y_pred = model.predict(train_test_data['X_test'])
        accuracy = accuracy_score(train_test_data['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 4),
            'model': 'Decision Tree'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train_ada_boost', methods=['POST'])
def train_ada_boost():
    try:
        data = request.get_json()
        n_estimators = int(data.get('n_estimators', 50))

        model = AdaBoostClassifier(n_estimators=n_estimators)
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'AdaBoost'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_svm', methods=['POST'])
def train_svm():
    try:
        data = request.get_json()
        kernel = data.get('kernel', 'rbf')
        c_param = float(data.get('c_param', 1.0))
        gamma = data.get('gamma', 'scale')

        model = SVC(kernel=kernel, C=c_param, gamma=gamma)
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'SVM'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_bagging', methods=['POST'])
def train_bagging():
    try:
        data = request.get_json()
        n_estimators = int(data.get('n_estimators', 10))
        max_samples = float(data.get('max_samples', 1.0))
        max_features = float(data.get('max_features', 1.0))

        model = BaggingClassifier(
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features
        )
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'Bagging'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_random_forest', methods=['POST'])
def train_random_forest():
    try:
        data = request.get_json()
        rf_criterion = data.get('rf_criterion', 'gini')
        rf_estimators = int(data.get('rf_estimators', 100))
        rf_max_depth = int(data.get('rf_max_depth', None)) if data.get('rf_max_depth') else None
        rf_max_features = data.get('rf_max_features', 'sqrt')

        model = RandomForestClassifier(
            criterion=rf_criterion,
            n_estimators=rf_estimators,
            max_depth=rf_max_depth,
            max_features=rf_max_features
        )
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'Random Forest'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_xgboost', methods=['POST'])
def train_xgboost():
    try:
        data = request.get_json()
        num_boost_round = int(data.get('num_boost_round', 100))
        early_stopping = int(data.get('early_stopping', 10))

        model = XGBClassifier(n_estimators=num_boost_round)
        model.fit(
            app.config['X_train'],
            app.config['y_train'],
            early_stopping_rounds=early_stopping,
            eval_set=[(app.config['X_test'], app.config['y_test'])],
            verbose=False
        )
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'XGBoost'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train_stacking', methods=['POST'])
def train_stacking():
    try:
        data = request.get_json()
        stacking_cv = int(data.get('stacking_cv', 5))
        final_estimator = data.get('final_estimator', 'logistic')

        base_models = [
            ('dt', DecisionTreeClassifier()),
            ('svm', SVC(probability=True))
        ]

        if final_estimator == 'logistic':
            final_est = LogisticRegression()
        else:
            final_est = DecisionTreeClassifier()

        model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_est,
            cv=stacking_cv
        )
        model.fit(app.config['X_train'], app.config['y_train'])
        y_pred = model.predict(app.config['X_test'])
        accuracy = accuracy_score(app.config['y_test'], y_pred)

        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'model': 'Stacking'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('uploaded_file', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
