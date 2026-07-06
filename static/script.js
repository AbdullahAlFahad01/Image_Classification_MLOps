async function predict() {

    let file = document.getElementById("image").files[0];

    let formData = new FormData();

    formData.append("file", file);

    let response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let data = await response.json();

    document.getElementById("result").innerHTML =
        "Prediction : " + data["Predicted Class"] +
        "<br><br>" +
        "Confidence : " + data["Confidence"] + "%";
}