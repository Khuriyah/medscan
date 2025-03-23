// document.addEventListener("DOMContentLoaded", function() {
//     fetch("/api/gejala", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" }
//     })
//     .then(response => response.json())
//     .then(data => {
//         let datalist = document.getElementById("gejalaList");
//         data.forEach(gejala => {
//             let option = document.createElement("option");
//             option.value = gejala;
//             datalist.appendChild(option);
//         });
//     });
// });

function checkSymptom() {
    const gejala = document.getElementById("gejalaInput").value;
    if (!gejala) {
        document.getElementById("result").innerHTML = "<p style='color:yellow;'>⚠️ Masukkan gejala terlebih dahulu!</p>";
        return;
    }
    
    fetch("api/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ gejala: gejala })
    })
    .then(response => response.json())
    .then(data => {
        let output = `<p><i class='fas fa-pills'></i> <strong>Obat:</strong> ${data.obat}</p>`;
        document.getElementById("result").innerHTML = output;
    })
    .catch(error => {
        document.getElementById("result").innerHTML = `<p style='color:red;'>⚠️ Terjadi kesalahan, coba lagi!</p>`;
    });
}
