from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles  # Import pro statické soubory
from porcupaine.porcupaine_score import compute_porcupaine_score
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Specify the templates directory
templates = Jinja2Templates(directory="templates")

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
    pai_score = compute_porcupaine_score(name, description, public_interest, district, category, budget)
    # Process the form data as needed
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "name": name,
            "description": description,
            "public_interest": public_interest,
            "district": district,
            "category": category,
            "budget": budget,
            "pai_score": pai_score
        },
    )

# Run the app with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
