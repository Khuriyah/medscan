from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Dataset langsung dalam kode (tanpa .csv)
data = {
    "gejala": [
        "Demam", "Batuk Kering", "Sakit Tenggorokan", "Alergi (Pilek)", "Nyeri Otot",
        "Anemia", "Sakit Kepala", "Mual dan Muntah", "Diare", "Ruam Kulit", "Maag",
        "Sesak Napas", "Sariawan", "Kram Otot", "Kesemutan", "Gangguan Tidur",
        "Kehilangan Nafsu Makan", "Sakit Punggung", "Nyeri Sendi", "Sakit Gigi",
        "Nyeri Perut", "Gatal-Gatal", "Vertigo", "Nyeri Ulu Hati", "Nyeri Haid",
        "Mata Gatal dan Merah", "Bibir Pecah-Pecah", "Telinga Berdenging",
        "Susah BAB", "Batuk Berdahak"
    ],
    "obat": [
        "Paracetamol, Panadol, Fasidol Forte, Bufect Suspensi",
        "Actifed Plus, Sanadryl Dmp, Siladex Antitusive, Vicks Formula 44, Levosif",
        "Lozenges, Obat Kumur Antiseptik",
        "CTM 4 mg, Cetirizine 10 mg",
        "Neo Rheumacyl, Counterpain, Salonpas Koyo",
        "Sakatonik Activ, Sangobion, Maltofer",
        "Biogesic, Panadol, Bodrex Extra",
        "Polysilane, Primperan, Domperidone",
        "Diatabs, Lodia, Entrostop",
        "Krim Hidrokortison, Krim Benoson, Krim Cinogenta",
        "Mylanta, Aciblok 150 mg",
        "Neo Napacin, Lasmalin, Lasal 2 mg",
        "Kuldon Sariawan",
        "Voltaren Gel, Salonpas Koyo, Oskadon SP",
        "Mecobalamin, Provital, Dolo-Neurobion",
        "Sedares 25 mg",
        "Curcuma Force, Curcuma Plus, Curvit",
        "Counterpain PXM Gel, Salonpas Gel",
        "Kaltrofen, Salonpas Hot Krim",
        "Asam Mefenamat, Cataflam, Aloclair Plus",
        "Spasminal, Feminax",
        "Ozen 10 mg, Claritin 10 mg",
        "Betahistine 6 mg, Mertigo SR",
        "Mylanta, Polysilane",
        "Spedifen 400 mg, Farsifen 400 mg",
        "Insto Reguler, Rohto Dry Fresh",
        "Vaseline Repairing Jelly",
        "Vital Ear Drops",
        "Dulcolax 5 mg",
        "Bisolvon Extra, Sanadryl Expectorant"
    ]
}

# DataFrame untuk pencocokan
df = pd.DataFrame(data)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["gejala"])

# Fungsi pencocokan gejala
def symptom_checker(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()
    return {
        "gejala": df.iloc[index]["gejala"],
        "obat": df.iloc[index]["obat"]
    }

# Endpoint halaman utama
@app.route('/')
def index():
    return render_template("index.html")

# Endpoint API untuk menerima input gejala
@app.route('/api/gejala', methods=['POST'])
@app.route('/api/check', methods=['POST'])
def check():
    data = request.get_json()
    gejala_input = data.get("gejala", "")
    if not gejala_input:
        return jsonify({"error": "Gejala tidak boleh kosong"}), 400
    result = symptom_checker(gejala_input)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
