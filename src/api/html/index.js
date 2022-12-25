const imagesDiv = document.getElementById("images");
const imgTemplate = document.getElementById("imgTemplate");
const rows = document.getElementById("rowNumbers");
const resTemplate = document.getElementById("resTemplate");
const charTemplate = document.getElementById("charTemplate")
let files = []

document.getElementById("fileupload")
    .addEventListener("change", event => files = [...event.target.files]);

document.getElementById("submit")
    .addEventListener("click", ev => {
        files.forEach( file => {
            const formdata = new FormData();
            formdata.append("file", file)
            fetch("/api/image/"+rows.value, {method: "Post", body: formdata})
                .then( res => {return res.json()})
                 .then( data => {
                     const filename = data.filename;
                     console.log(data)
                     const copy = imgTemplate.content.cloneNode(true);
                     const img = copy.getElementById("img");
                     img.src = "/api/images/"+filename+".png";
                     img.alt = filename
                     data.result.forEach( letters => {
                         copy.getElementById("text").innerHTML += "<p>" + letters.join(" - ") + "</p>";
                     } );
                     copy.getElementById("heading").textContent += filename;
                     imagesDiv.appendChild(copy);
                 }
            );
        });
    });


document.getElementById("getRes")
    .addEventListener("click", ev => {
        fetch("/api/res")
            .then(res => res.json())
            .then(
                data => {
                    const copy = resTemplate.content.cloneNode(true);
                    const container = copy.getElementById("container");
                    for(let c in data){
                        const charCopy = charTemplate.content.cloneNode(true);
                        charCopy.getElementById("charH").textContent += c;
                        charCopy.getElementById("all").textContent += data[c]["all"];
                        charCopy.getElementById("right").textContent += data[c]["right"];
                        charCopy.getElementById("percentage").textContent +=
                            data[c]["percentage"].toFixed(4);

                        container.appendChild(charCopy);
                    }
                    imagesDiv.appendChild(copy);
                }
            )
    });