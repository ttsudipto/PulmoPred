<!DOCTYPE html>
<html>
    <head>
        <title>Add Pulmonary Function Report Details</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type = "text/javascript" src = "js/plot.js"></script>
        <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-csv/1.0.8/jquery.csv.min.js"></script>
        <script type = "text/javascript" src = "https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script type = "text/javascript" src = "js/plot.js"></script>
    </head>
    <body>
        <div class = "section_header">
            <center> <h1> Heading </h1> </center>
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
            <h3>Result :</h3>
            <table class = "form" border = "1" id = "rtable">
                <tr>
                    <th>Classifiers</th>
                    <th>Predicted score</th>
                    <th>Threshold</th>
                    <th>Predicted class</th>
                </tr>
                <tr>
                    <td><b>Classifier 0</b></td>
                    <td><?php echo round(floatval($result->score0), 2); ?></td>
                    <td><?php echo $result->threshold0; ?></td>
                    <td><?php echo $result->predicted_label0; ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 1</b></td>
                    <td><?php echo round(floatval($result->score1), 2); ?></td>
                    <td><?php echo $result->threshold1; ?></td>
                    <td><?php echo $result->predicted_label1; ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 2</b></td>
                    <td><?php echo round(floatval($result->score2), 2); ?></td>
                    <td><?php echo $result->threshold2; ?></td>
                    <td><?php echo $result->predicted_label2; ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 3</b></td>
                    <td><?php echo round(floatval($result->score3), 2); ?></td>
                    <td><?php echo $result->threshold3; ?></td>
                    <td><?php echo $result->predicted_label3; ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 4</b></td>
                    <td><?php echo round(floatval($result->score4), 2); ?></td>
                    <td><?php echo $result->threshold4; ?></td>
                    <td><?php echo $result->predicted_label4; ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 5</b></td>
                    <td><?php echo round(floatval($result->score5), 2); ?></td>
                    <td><?php echo $result->threshold5; ?></td>
                    <td><?php echo $result->predicted_label5; ?></td>
                </tr>
            </table>
            <?php
                $scores = array($result->score0, $result->score1, $result->score2, $result->score3, $result->score4, $result->score5);
                $thresholds = array($result->threshold0, $result->threshold1, $result->threshold2, $result->threshold3, $result->threshold4, $result->threshold5);
            ?>
            <center>
            <table border = "1" id = "dtable">
                <tr>
                    <td><div id="plt_div0" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div0',0,".$scores[0].",".$thresholds[0].");</script>"; ?></div></td>
                    <td><div id="plt_div1" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div1',1,".$scores[1].",".$thresholds[1].");</script>"; ?></div></td>
                    <td><div id="plt_div2" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div2',2,".$scores[2].",".$thresholds[2].");</script>"; ?></div></td>
                </tr>
                <tr>
                    <td><div id="plt_div3" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div3',3,".$scores[3].",".$thresholds[3].");</script>"; ?></div></td>
                    <td><div id="plt_div4" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div4',4,".$scores[4].",".$thresholds[4].");</script>"; ?></div></td>
                    <td><div id="plt_div5" style="width:400px; height:300px;"><?php echo "<script>get_density('plt_div5',5,".$scores[5].",".$thresholds[5].");</script>"; ?></div></td>
                </tr>
            </table>
            </center>
            <?php
                    echo count($out)."<br/>";
                    for($i=0;$i<count($out);++$i)
                        echo $out[$i]."<br/>";
                    echo $status;
                }
            ?>
        </div>
    </body>
</html> 
