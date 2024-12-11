<<<<<<< HEAD
// Zobrazení textového pole při výběru "jiný"
function toggleCustomCategory() {
    const categorySelect = document.getElementById('category');
    const customCategoryDiv = document.getElementById('customCategoryDiv');
}

// Načítání dat z JSON souboru
fetch('/static/data.json')
    .then(response => response.json())
    .then(data => {
        // Načtení kategorií
        const categorySelect = document.getElementById('category');
        data.categories.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            categorySelect.appendChild(opt);
        });

        // Načtení městských částí
        const districtSelect = document.getElementById('district');
        data.districts.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            districtSelect.appendChild(opt);
        });
    })
    .catch(error => console.error('Chyba při načítání dat:', error));
=======
// Zobrazení textového pole při výběru "jiný"
function toggleCustomCategory() {
    const categorySelect = document.getElementById('category');
    const customCategoryDiv = document.getElementById('customCategoryDiv');
    const selectedValue = categorySelect.value;

    // Zobrazit/skryt textové pole
    if (selectedValue === "jiny") {
        customCategoryDiv.style.display = "block";
    } else {
        customCategoryDiv.style.display = "none";
    }
}

// Načítání dat z JSON souboru
fetch('/static/data.json')
    .then(response => response.json())
    .then(data => {
        // Načtení kategorií
        const categorySelect = document.getElementById('category');
        data.categories.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            categorySelect.appendChild(opt);
        });

        // Načtení městských částí
        const districtSelect = document.getElementById('district');
        data.districts.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.label;
            districtSelect.appendChild(opt);
        });
    })
    .catch(error => console.error('Chyba při načítání dat:', error));
>>>>>>> eab99781777f12a6ffd762707f3e13420aa32cbf
