import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)

model=load_model("fruit.h5")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)
        index=['Apple___Black_rot','Apple___healthy','Corn_(maize)___healthy','Corn_(maize)___Northern_Leaf_Blight','Peach___Bacterial_spot','Peach___healthy']
        if(pred[0]==0):
            text="The Classified Plant Disease is : " +str(index[pred[0]] + ". Fertilizers recommended are Captan and fungicides containing a strobulurin (FRAC Group 11 Fungicides) as an active ingredient are effective controlling black rot on fruit.")
        elif(pred[0]==3):
            text="The Classified Plant Disease is : " +str(index[pred[0]] + ". Fertilizers recommended are Bio-fungicides based on Trichoderma harzianum, or Bacillus subtilis can be applied at different stages to decrease the risk of infection. Application of sulfur solutions is also effective.")
        elif(pred[0]==4):
            text="The Classified Plant Disease is : " +str(index[pred[0]] + ". Fertilizers recommended are Copper-based sprays alone or together with an antibiotic can be used preventively with moderate efficacy. Dosage must be reduced progressively to avoid damage to leaves.")
        else:
            text="The Classified Plant Disease is : " +str(index[pred[0]] + ". And the plant is healthy.")
    return text

if __name__=='__main__':
    app.run(debug=False)