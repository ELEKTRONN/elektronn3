from flask import Flask, send_file, make_response
from plotting_flask import do_plot
app = Flask(__name__)

@app.route('/', methods=['GET'])
def check_knossos():
    bytes_obj = do_plot()
    
    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False)
