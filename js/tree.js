
function getTree(model, paths, version) {
    var treeIndex = parseInt(document.getElementById("tree"+model).value);
    var params = "model=" + model + "&tree=" + treeIndex + "&path=" + JSON.stringify(paths[model][treeIndex]) + "&version=" + version;
    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            var msg = version + " - " + model + " - " + treeIndex + " - [" + paths[model][treeIndex] + "]";
            document.getElementById('tree_container').innerHTML = this.responseText;// + "<p>" + msg + "</p>";
            document.getElementById('tree_container').style.borderTop = "1px solid black";
            document.getElementById('tree_container').style.borderBottom = "1px solid black";
        }
    };
    xhttp.open("POST", "tree.php", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send(params);
}

// function zoomSVG() {
//     var svgElement = document.getElementsByTagName('svg')[0];
//     var width = parseInt(svgElement.getAttribute('width').split('%')[0]);
//     var height = parseInt(svgElement.getAttribute('height').split('%')[0]);
//     svgElement.setAttribute('width', (width+10)+'%');
//     svgElement.setAttribute('height', (height+10)+'%');
// }
