<?php
    function readSVG($model, $tree, $version) {
        $filename = "output/graphs/".$version."svg/".$model."/".$tree.".svg";
        $xml = simplexml_load_file($filename);
        return $xml;
    }
    
    function modifyDimensions(&$svg) {
        $svg['width'] = "100%";
        $svg['height'] = "100%";
        $svg->g->polygon['fill'] = "#fff5e6";
    }
    
    function plotPath(&$svg, $path) {
        $elements = $svg->g->g;
        for($i=0; $i<count($path); ++$i) {
            foreach ($svg->g->g as $element) {
                if ($element->title == (string)$path[$i] && $element['class'] == "node")
                    $element->polygon['fill'] = "#a1ddc0";
                if ($i+1<count($path))
                    if ($element->title == $path[$i]."->".$path[$i+1] && $element['class'] == "edge") {
                        $element->polygon['fill'] = "red";
                        $element->polygon['stroke'] = "red";
                        $element->path['stroke'] = "red";
                    }
            }
        }
    }
    
    $model = intval($_POST["model"]);
    $tree = intval($_POST["tree"]);
    $version = $_POST["version"];//"0.20.3";
    $path = json_decode($_POST["path"]);
    
    $svg = readSVG($model, $tree, $version);
    modifyDimensions($svg);
    plotPath($svg, $path);
    
    echo "<center><h3>Classifier: " . $model . " - Decision Tree: " . $tree . "</h3>";
//     echo "<<h5>Path : [ ";
//     for($i=0; $i<count($path); ++$i) {
//         echo $path[$i] . "&nbsp";
//     }
//     echo "]</h5></center>";
    echo $svg->asXML();
    
?>
