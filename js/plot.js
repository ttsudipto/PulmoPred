function get_density(div_id, index, score, threshold)
{
    var xmlhttp = new XMLHttpRequest();
    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            plot_density(div_id, this.responseText, score, threshold);
        }
    };
    xmlhttp.open('GET', 'output/densities/'+index+'.csv', true);
    xmlhttp.send();
}

function get_column(csv, index)
{
    var column = []
    for(var i=0; i<csv.length; ++i)
    {
        column[i] = csv[i][index];
    }
    return column;
}

function plot_one(div_id, pos_x, pos_y, neg_x, neg_y, score, threshold)
{
    var graphDiv = document.getElementById(div_id);
    var pos_curve = {
        x: pos_x,
        y: pos_y,
        name: 'Positive density',
        line: {width: 1}
    };
    var neg_curve = {
        x: neg_x,
        y: neg_y,
        name: 'Negative density',
        line: {width: 1}
    };
    var score_pt = {
        mode: 'markers',
        x: [score],
        y: [0],
        name: 'Calculated score'
    }
    var data = [pos_curve, neg_curve, score_pt];
    var layout = {
        xaxis: {
            visible : true,
            title : {text : 'SVM scores'}
        },
        yaxis: {
            visible : true,
            title : {text : 'Probability'}
        },
        shapes: [
            {
                type: 'line',
                yref: 'paper',
                x0: threshold,
                y0: 0,
                x1: threshold,
                y1: 1,
                name: 'Threshold',
                line:{
                    dash:'dot'
                }
            }
        ],
        plot_bgcolor: '#ffe6cc',//'#e6e6e6',
        paper_bgcolor: '#ffe6cc',//'#e6e6e6',
        margin: {t:0}
    };

    Plotly.plot(graphDiv, data, layout, {showSendToCloud:true});
}

function plot_density(div_id, response, score, threshold)
{
    var csv = $.csv.toArrays(response);
    
    var pos_x = get_column(csv, 0);
    var pos_y = get_column(csv, 1);
    var neg_x = get_column(csv, 2);
    var neg_y = get_column(csv, 3);
    
    plot_one(div_id, pos_x, pos_y, neg_x, neg_y, score, threshold);
}
