<!DOCTYPE html>
<html>
    <head>
        <title>Add Pulmonary Function Report Details</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type = "text/javascript" src = "js/plot.js"></script>
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-csv/1.0.8/jquery.csv.min.js"></script>
        <script type = "text/javascript" src = "https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <style>
            * {
                box-sizing: border-box;
            }
            .plot_container::after {
                content: "";
                clear: both;
                display: table;
            }
            .plot_div {
                width: 100%;
                height: 300px;
            }
            .plot_caption {
                width: 100%;
                text-align: center;
                background-color: #cce6ff;
                font-weight: bold;
            }
        </style>
        <div class = "section_header">
            <table border="0" cellpadding="0" style="width:100%;">
                <tr>
                    <td style="width:30%;background-color:#155151;border-right:2px solid blue;color:white;"><center><h1>PulmoPred</h1></center></td>
                    <td style="width:70%;background-color:#c2d6d6;border-left:2px solid blue;"><center><h2>Predict obstructive and non-obstructive pulmonary diseases using spirometry</h2></center></td>
                </tr>
            </table>
        </div>

        <div class = "section_menu">
            <table>
                <tr class="nav">
                    <td class="nav"><a href="#" class="side_nav">Home</a></td>
                    <td class="nav"><a href="#" class="active">Classification</a></td>
                </tr>
            </table>
        </div>

        <div class = "section_left">
        </div>
        
        <div class = "section_middle">
            <form action = "index.php" method = "POST">
                <table class = "form" border = "0" id = "stable">
                    <tr>
                        <th rowspan = "2">Measurements</th>
                        <th colspan = "2">Pre</th>
                        <th colspan = "2">Post</th>
                    </tr>
                    <tr>
                        <th>Value</th>
                        <th>Pred %</th>
                        <th>Value</th>
                        <th>Pred %</th>
                    </tr>
                    <tr>
                        <td style="padding-left : 5px;">
                                FEV1 - Forced Expiratory Volume
                        </td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fev1_pre_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fev1_pre_percent" /></td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fev1_post_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fev1_post_percent" /></td>
                    </tr>
                    <tr>
                        <td style="padding-left : 5px;">
                                FVC - Forced Vital Capacity
                        </td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fvc_pre_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fvc_pre_percent" /></td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fvc_post_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fvc_post_percent" /></td>
                    </tr>
                    <tr>
                        <td style="padding-left : 5px;">
                                FEF 25-75% - Forced Expiratory Flow
                        </td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fef_pre_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fef_pre_percent" /></td>
                        <td><input type = "number" class = "full" min = "0" step = "0.001" name = "fef_post_value" /></td>
                        <td><input type = "number" class = "full" min = "0" name = "fef_post_percent" /></td>
                    </tr>
                </table>
                <br/>
                <table class = "form" border = "0">
                    <tr>
                        <th>Classification model</th>
                        <td><center><input type = "radio" name = "model_id" value = "SVM" checked /><br/>Support Vector Machine (SVM)</center></td>
                        <td><center><input type = "radio" name = "model_id" value = "RF"/><br/>Random Forest (RF)</center></td>
                        <td><center><input type = "radio" name = "model_id" value = "GNB"/><br/>Naive Bayes (NB)</center></td>
                    </tr>
                </table>
                <input type = "hidden" id = "query" name = "query" value = "0" />
                <br/>
                <center>
                    <input type = "submit" value = "Submit" onclick = "document.getElementById('query').value = '1';"/>
                </center>
            </form>
            
            <?php
                if(array_key_exists("query", $_POST) && $_POST["query"] == "1")
                {
                    $params = array();
                    $params["fev1_pre_value"] = $_POST["fev1_pre_value"];
                    $params["fev1_pre_percent"] = $_POST["fev1_pre_percent"];
                    $params["fev1_post_value"] = $_POST["fev1_post_value"];
                    $params["fev1_post_percent"] = $_POST["fev1_post_percent"];
                    $arg_json = json_encode($_POST);
//                     echo $arg_json;
                    $command = "venv/bin/python -m python.driver '".$arg_json."' 2>&1";
//                     echo "<pre>".$command."</pre>\n";
                    exec($command, $out, $status);
                    $result = json_decode($out[0]);
//                     print_r($result);
            ?>
            <h3>Input :</h3>    
            <table class = "form" border = "0" cellpadding="5px" id = "stable">
                <tr>
                    <th rowspan = "2">Measurements</th>
                    <th colspan = "2">Pre</th>
                    <th colspan = "2">Post</th>
                </tr>
                <tr>
                    <th>Value</th>
                    <th>Pred %</th>
                    <th>Value</th>
                    <th>Pred %</th>
                </tr>
                <tr>
                    <td><center>FEV1 - Forced Expiratory Volume</center></td>
                    <td><center><?php echo $_POST["fev1_pre_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fev1_pre_percent"]; ?></center></td>
                    <td><center><?php echo $_POST["fev1_post_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fev1_post_percent"]; ?></center></td>
                </tr>
                <tr>
                    <td><center>FVC - Forced Vital Capacity</center></td>
                    <td><center><?php echo $_POST["fvc_pre_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fvc_pre_percent"]; ?></center></td>
                    <td><center><?php echo $_POST["fvc_post_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fvc_post_percent"]; ?></center></td>
                </tr>
                <tr>
                    <td><center>FEF 25-75% - Forced Expiratory Flow</center></td>
                    <td><center><?php echo $_POST["fef_pre_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fef_pre_percent"]; ?></center></td>
                    <td><center><?php echo $_POST["fef_post_value"]; ?></center></td>
                    <td><center><?php echo $_POST["fef_post_percent"]; ?></center></td>
                </tr>
                <tr>
                    <th>Classification model</th>
                    <td colspan="4">
                        <center>
                        <?php 
                            if($_POST["model_id"] == "SVM")
                                echo "Support Vector Machine (SVM)";
                            elseif ($_POST["model_id"] == "RF")
                                echo "Random Forest (RF)";
                            elseif ($_POST["model_id"] == "GNB")
                                echo "Naive Bayes (NB)";
                        ?>
                        </center>
                    </td>
                </tr>
            </table>
            
            <h3>Result :</h3>
                    
            <?php
                    if ($_POST["model_id"] == "SVM")
                    {
            ?>
                        <table class = "form" border = "0" cellpadding="5px" id = "rtable">
                            <tr>
                                <th>Classifiers</th>
                                <th>Predicted score</th>
                                <th>Threshold</th>
                                <th>Predicted class</th>
                                <th>Positiveness (%)</th>
                                <th>Negativeness (%)</th>
                            </tr>
                            <?php
                                $n_models = count($result->labels);
                                for($i=0; $i<$n_models; ++$i) {
                                    echo "<tr>";
                                    echo "<td><center><b>Classifier ".$i."</b></center></td>";
                                    echo "<td><center>".round(floatval($result->scores[$i]), 3)."</center></td>";
                                    echo "<td><center>".$result->thresholds[$i]."</center></td>";
                                    echo "<td><center>".$result->labels[$i]."</center></td>";
                                    echo "<td><center>".round(floatval($result->positivenesses[$i]), 4)."</center></td>";
                                    echo "<td><center>".round(floatval($result->negativenesses[$i]), 4)."</center></td>";
                                    echo "</tr>";
                                }
                            ?>
                        </table>
                        <br/><br/>
                        <div class="plot_container">
                            <div style="width:49%; float:left;">
                                <div id="plt_div0" class="plot_div"><?php echo "<script>get_density('plt_div0',0,".$result->scores[0].",".$result->thresholds[0].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 0</div><br/>
                                <div id="plt_div2" class="plot_div"><?php echo "<script>get_density('plt_div2',2,".$result->scores[2].",".$result->thresholds[2].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 2</div><br/>
                                <div id="plt_div4" class="plot_div"><?php echo "<script>get_density('plt_div4',4,".$result->scores[4].",".$result->thresholds[4].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 4</div><br/>
                            </div>
                            <div style="width:49%; float:right;">
                                <div id="plt_div1" class="plot_div"><?php echo "<script>get_density('plt_div1',1,".$result->scores[1].",".$result->thresholds[1].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 1</div><br/>
                                <div id="plt_div3" class="plot_div"><?php echo "<script>get_density('plt_div3',3,".$result->scores[3].",".$result->thresholds[3].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 3</div><br/>
                                <div id="plt_div5" class="plot_div"><?php echo "<script>get_density('plt_div5',5,".$result->scores[5].",".$result->thresholds[5].");</script>"; ?></div>
                                <div class="plot_caption">Classifier 5</div><br/>
                            </div>
                        </div>
            <?php
                    } elseif ($_POST["model_id"] == "RF") {
            ?>
                        <script>
                            var paths_json = "<?php echo json_encode($result->paths); ?>";
                            var paths = JSON.parse(paths_json);
                            var version = "<?php echo $result->version; ?>"
                        </script>
                        <script type = "text/javascript" src = "js/tree.js"></script>
                        
                        <table class = "form" border = "0" cellpadding="5px" id = "rtable">
                            <tr>
                                <th>Classifiers</th>
                                <th>Predicted class</th>
                                <th>Probability</th>
                                <th>Decision Trees</th>
                            </tr>
                            <?php
                                $n_models = count($result->labels);
                                for($i=0; $i<$n_models; ++$i) {
                                    echo "<tr>";
                                    echo "<td><center><b>Classifier ".$i."</b></center></td>";
                                    echo "<td><center>".$result->labels[$i]."</center></td>";
                                    echo "<td><center>".round(floatval($result->probas[$i]), 3)."</center></td>";
                                    echo "<td><center><select id=\"tree".$i."\" name=\"tree".$i."\">";
                                    for($j=0; $j<count($result->paths[$i]); ++$j)
                                        echo "<option value=\"".$j."\">Tree-".$j."</option>";
                                    echo "</select>&nbsp;<button onclick=\"getTree(".$i.", paths, version);\">Get</button></center></td>";
                                    echo "</tr>";
                                }
                            ?>
                        </table>
                        <br/><br/>
                        <div class="plot_container" id="tree_container" style="background-color:#ffebcc; width:100%; overflow:auto;"></div>
            <?php
                    } elseif ($_POST["model_id"] == "GNB") {
            ?>
                        <table class = "form" border = "0" cellpadding="5px" id = "rtable">
                            <tr>
                                <th>Classifiers</th>
                                <th>Predicted class</th>
                                <th>Probability</th>
                            </tr>
                            <?php
                                $n_models = count($result->labels);
                                for($i=0; $i<$n_models; ++$i) {
                                    echo "<tr>";
                                    echo "<td><center><b>Classifier ".$i."</b></center></td>";
                                    echo "<td><center>".$result->labels[$i]."</center></td>";
                                    echo "<td><center>".round(floatval($result->probas[$i]), 5)."</center></td>";
                                    echo "</tr>";
                                }
                            ?>
                        </table>
                        <br/><br/>
            <?php
                    }
                    echo "<br/><div>";
                    echo count($out)."<br/>";
                    for($i=0;$i<count($out);++$i)
                        echo $out[$i]."<br/>";
                    echo $status;
                    echo "</div>";
                }
            ?>
            
            </div>
        </div>
    </body>
</html> 
