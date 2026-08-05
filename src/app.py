from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.policies.thompson_sampling import ThompsonSampling

# carrega os dados
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "synthetic_enrichment"

offer_events = pd.read_csv(DATA_DIR / "offer_events.sample.csv", parse_dates=["event_date"])

offer_catalog = pd.read_csv(DATA_DIR / "offer_catalog.sample.csv")

# inicia e treina o Thompson utilizando histórico sintético 
thompson = ThompsonSampling(random_state=42)
thompson.run(offer_events)

# configuração da API
app = FastAPI(
    title="Offer Recommendation API",
    description="API de recomendação de ofertas utilizando Contextual Thompson Sampling",
    version="1.0.0"
)

# modelo de entrada com contexto necessário do cliente
class Customer(BaseModel):
    age: int
    job: str
    education: str
    poutcome: str

# interface simples para demo
@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>

<html>

<head>

<title>Offer Recommendation</title>

<style>

body{
    font-family: Arial;
    margin: 20px;
    background: #f5f5f5;
}

.container{
    background: white;
    max-width: 700px;
    margin: auto;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,.15);
}

h2{
    text-align: center;
    margin-bottom: 20px;
    font-size: 24px;
}

label{
    font-weight: bold;
    display: block;
    margin-bottom: 4px;
}

input,
select{
    width: 100%;
    padding: 6px;
    margin-bottom: 10px;
    box-sizing: border-box;
}

button{
    width: 100%;
    padding: 10px;
    background: #1f77b4;
    color: white;
    border: none;
    font-size: 15px;
    cursor: pointer;
    margin-top: 5px;
}

button:hover{
    background: #155d8b;
}

#result{
    margin-top: 15px;
    padding: 12px;
    background: #eeeeee;
    border-radius: 5px;
    line-height: 1.5;
}

</style>

</head>

<body>

<div class="container">

<h2>Offer Recommendation - Contextual Thompson Sampling</h2>

<label>Idade</label>
<input id="age" type="number">

<label>Cargo</label>
<select id="job">
<option value="" selected disabled>Selecione um cargo</option>
<option>admin.</option>
<option>blue-collar</option>
<option>entrepreneur</option>
<option>housemaid</option>
<option>management</option>
<option>retired</option>
<option>self-employed</option>
<option>services</option>
<option>student</option>
<option>technician</option>
<option>unemployed</option>
<option>unknown</option>
</select>

<label>Escolaridade</label>
<select id="education">
<option value="" selected disabled>Selecione a escolaridade</option>
<option>basic.4y</option>
<option>basic.6y</option>
<option>basic.9y</option>
<option>high.school</option>
<option>illiterate</option>
<option>professional.course</option>
<option>university.degree</option>
<option>unknown</option>
</select>

<label>Campanha anterior</label>
<select id="poutcome">
<option value="" selected disabled>Selecione o resultado da campanha anterior</option>
<option>success</option>
<option>failure</option>
<option>nonexistent</option>
</select>

<button onclick="recommend()">
Recomendar oferta
</button>

<div id="result">
Sem recomendações.
</div>

</div>

<script>

async function recommend(){

    const response = await fetch("/recommend",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({
            age:Number(document.getElementById("age").value),
            job:document.getElementById("job").value,
            education:document.getElementById("education").value,
            poutcome:document.getElementById("poutcome").value
        })

    });

    const data = await response.json();

    document.getElementById("result").innerHTML=
    "<b>Recommended Arm:</b> <br>" +
    data.recommendation.arm_id + " - " + data.recommendation.offer_name + "<br><br>" +
    "<b>Message:</b><br>"+data.recommendation.message;

}

</script>

</body>

</html>
"""

# endpoint para consumo
@app.post("/recommend")
def recommend(customer: Customer):

    arm = thompson.recommend_arm(customer.model_dump())

    offer = (offer_catalog[offer_catalog["arm_id"] == arm].iloc[0])

    return {
        "customer": customer.model_dump(),
        "recommendation":{
            "arm_id": int(offer["arm_id"]),
            "offer_name": offer["arm_name"],
            "message": offer["message"]
        }
    }