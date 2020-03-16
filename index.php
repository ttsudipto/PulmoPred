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
                    <th>Positiveness (%)</th>
                    <th>Negativeness (%)</th>
                </tr>
                <tr>
                    <td><b>Classifier 0</b></td>
                    <td><?php echo round(floatval($result->score0), 3); ?></td>
                    <td><?php echo $result->threshold0; ?></td>
                    <td><?php echo $result->predicted_label0; ?></td>
                    <td><?php echo round(floatval($result->positiveness0), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness0), 4); ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 1</b></td>
                    <td><?php echo round(floatval($result->score1), 3); ?></td>
                    <td><?php echo $result->threshold1; ?></td>
                    <td><?php echo $result->predicted_label1; ?></td>
                    <td><?php echo round(floatval($result->positiveness1), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness1), 4); ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 2</b></td>
                    <td><?php echo round(floatval($result->score2), 3); ?></td>
                    <td><?php echo $result->threshold2; ?></td>
                    <td><?php echo $result->predicted_label2; ?></td>
                    <td><?php echo round(floatval($result->positiveness2), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness2), 4); ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 3</b></td>
                    <td><?php echo round(floatval($result->score3), 3); ?></td>
                    <td><?php echo $result->threshold3; ?></td>
                    <td><?php echo $result->predicted_label3; ?></td>
                    <td><?php echo round(floatval($result->positiveness3), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness3), 4); ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 4</b></td>
                    <td><?php echo round(floatval($result->score4), 3); ?></td>
                    <td><?php echo $result->threshold4; ?></td>
                    <td><?php echo $result->predicted_label4; ?></td>
                    <td><?php echo round(floatval($result->positiveness4), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness4), 4); ?></td>
                </tr>
                <tr>
                    <td><b>Classifier 5</b></td>
                    <td><?php echo round(floatval($result->score5), 3); ?></td>
                    <td><?php echo $result->threshold5; ?></td>
                    <td><?php echo $result->predicted_label5; ?></td>
                    <td><?php echo round(floatval($result->positiveness5), 4); ?></td>
                    <td><?php echo round(floatval($result->negativeness5), 4); ?></td>
                </tr>
            </table>
            <br/><br/>
            <?php
                $scores = array($result->score0, $result->score1, $result->score2, $result->score3, $result->score4, $result->score5);
                $thresholds = array($result->threshold0, $result->threshold1, $result->threshold2, $result->threshold3, $result->threshold4, $result->threshold5);
            ?>
            <div class="plot_container">
                <div style="width:49%; float:left;">
                    <div id="plt_div0" class="plot_div"><?php echo "<script>get_density('plt_div0',0,".$scores[0].",".$thresholds[0].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 0</div><br/>
                    <div id="plt_div2" class="plot_div"><?php echo "<script>get_density('plt_div2',2,".$scores[2].",".$thresholds[2].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 2</div><br/>
                    <div id="plt_div4" class="plot_div"><?php echo "<script>get_density('plt_div4',4,".$scores[4].",".$thresholds[4].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 4</div><br/>
                </div>
                <div style="width:49%; float:right;">
                    <div id="plt_div1" class="plot_div"><?php echo "<script>get_density('plt_div1',1,".$scores[1].",".$thresholds[1].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 1</div><br/>
                    <div id="plt_div3" class="plot_div"><?php echo "<script>get_density('plt_div3',3,".$scores[3].",".$thresholds[3].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 3</div><br/>
                    <div id="plt_div5" class="plot_div"><?php echo "<script>get_density('plt_div5',5,".$scores[5].",".$thresholds[5].");</script>"; ?></div>
                    <div class="plot_caption">Classifier 5</div><br/>
                </div>
            </div>
            <?php
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
