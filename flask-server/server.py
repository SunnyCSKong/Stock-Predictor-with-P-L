from flask import Flask,jsonify, request,send_file
from flask_cors import CORS, cross_origin
from generatePredictions import generate_predictions
import csv
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import os

# Set Matplotlib backend to Agg (non-interactive)
matplotlib.use('Agg')

app = Flask(__name__)
CORS(app)
plot_directory = os.path.join(os.path.dirname(__file__))

@app.route('/listings')
def all_listings():
    # csv file name
    filename = "tickers.csv"

    # initializing the titles and rows list
    fields = []
    rows = []

    # reading csv file
    with open(filename, 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        # extracting field names through first row
        fields = next(csvreader)
        # extracting each data row one by one
        for row in csvreader:
            symbol = row[0]
            name = row[1]
            rows.append({"symbol":symbol,"name":name})
    return jsonify(rows)


@app.route('/submit_form', methods=['POST'])
@cross_origin(origin='*')
def submit_form():
    data = request.get_json()
    search_term = data.get("searchTerm")  
    average_type = data.get("average")
    plots_and_predictions = generate_predictions(search_term, average_type=average_type)
    return jsonify(plots_and_predictions)


@app.route('/get_plot/<plot_name>')
@cross_origin(origin='*')
def get_plot(plot_name):
    plot_path = os.path.join(plot_directory, plot_name)
    return send_file(plot_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
