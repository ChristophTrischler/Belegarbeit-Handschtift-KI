const imagesDiv = document.getElementById("images");
const template = document.getElementById("imgTemplate");
let files = []

document.getElementById("fileupload")
    .addEventListener("change", event => files = [...event.target.files] );

document.getElementById("submit")
    .addEventListener("click", ev => {
        files.forEach( file => {
            const formdata = new FormData();
            formdata.append("file", file)
            fetch("/api/image", {method: "Post", body: formdata})
                .then( res => {return res.json()})
                 .then( data => {
                     const filename = data.filename;
                     console.log(data)
                     const copy = template.content.cloneNode(true);
                     const img = copy.getElementById("img");
                     img.src = "/api/images/"+filename+".png";
                     img.alt = filename
                     copy.getElementById("text").textContent =  data.nums.join(" - ") +"=>"+ data.accuracy + "%";
                     copy.getElementById("heading").textContent += filename
                     imagesDiv.appendChild(copy)
                 }
            );
        });
    });