from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Import pro statické soubory
from porcupaine.porcupaine_score import compute_porcupaine_score
import uvicorn
import json  # Pro práci s JSON souborem

# Initialize the FastAPI app
app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Specify the templates directory
templates = Jinja2Templates(directory="templates")

# Načti data z JSON souboru
with open("static/data.json", encoding="utf-8") as f:
    data = json.load(f)

# Vytvoř slovníky pro vyhledávání
categories = {item["value"]: item["label"] for item in data["categories"]}
districts = {item["value"]: item["label"] for item in data["districts"]}

# Route for displaying the form
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

# Route for handling form submission
@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    public_interest: str = Form(...),
    district: str = Form(...),
    category: str = Form(...),
    budget: int = Form(...)
):
    # Nahrazení value za label
    district_label = districts.get(district, "Neznámá oblast")
    category_label = categories.get(category, "Neznámá kategorie")

    # Vypočítání porcupaine skóre
    pai_score = compute_porcupaine_score(name, description, public_interest, district, category, budget)

    # Předání popisků do šablony
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "name": name,
            "description": description,
            "public_interest": public_interest,
            "district": district_label,
            "category": category_label,
            "budget": budget,
            "pai_score": pai_score,
        },
    )


# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
