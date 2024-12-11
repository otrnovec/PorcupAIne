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
