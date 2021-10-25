var demoData = [
                    [1.12, 45, 1.6, 65, 1.76, 61, 2.22, 77, 0.81, 23, 1.27, 36], 
                    [2.07, 88, 1.93, 82, 2.47, 84, 2.37, 80, 2.39, 95, 2.09, 84],
               ];
function addDemoData(index) {
    var data = demoData[index];
    for(var i=0; i<data.length; ++i)
//                     alert('inp'+i+document.getElementById('inp'+i));
        document.getElementById('inp'+i).value = data[i];
}
