from flask import Flask, render_template, request,redirect
from help import preprocessing,vectorizer,get_prediction

app =Flask(__name__)

data =dict()
reviews = []
positive=0
negative=0

print("======================================Flask Server Started====================================")

@app.route("/")
def index():
    data['reviews'] = reviews
    data['positive'] = positive
    data['negative']= negative
    return render_template('index.html', data=data)

@app.route("/", methods=['post'])
def my_post():
    text = request.form['text']
    print(f'text :{text}')
    preprocessed_text = preprocessing(text);
    print(f'preprcoessed_text: {preprocessed_text}')
    vectorized_text = vectorizer(preprocessed_text)
    print(f'vectorized_text: {vectorized_text}')
    predictions = get_prediction(vectorized_text)
    print(f'Predictions: {predictions}')

    if predictions =="Negative":
        global negative
        negative += 1
    else:
        global positive
        positive += 1

    reviews.insert(0,text)
    return redirect(request.url)
if __name__ == 'main':
    app.run()
