<!DOCTYPE html>
<html>
    <head>
        <title>Add Pulmonary Function Report Details</title>
        <link rel = "stylesheet" type = "text/css" href = "css/main.css" />
        <script type = "text/javascript" src = "js/pft.js"></script>
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
    <!--                     <th></th> -->
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
                    echo "<div><ul>";
                    echo "<li>FEV1 Pre Value = ".$_POST['fev1_pre_value']."</li>";
                    echo "<li>FEV1 Pre Percent = ".$_POST['fev1_pre_percent']."</li>";
                    echo "<li>FEV1 Post Value = ".$_POST['fev1_post_value']."</li>";
                    echo "<li>FEV1 Post Percent =". $_POST['fev1_post_percent']."</li>";
                    echo "</ul></div>";
                    $params = array();
                    $params["fev1_pre_value"] = $_POST["fev1_pre_value"];
                    $params["fev1_pre_percent"] = $_POST["fev1_pre_percent"];
                    $params["fev1_post_value"] = $_POST["fev1_post_value"];
                    $params["fev1_post_percent"] = $_POST["fev1_post_percent"];
                    $arg_json = json_encode($_POST);
                    echo $arg_json."foo\n";
                    $command = "../../../../venv/bin/python python/site/driver.py '".$arg_json."' 2>&1";
                    echo "<pre>".$command."</pre>\n";
                    exec($command, $out, $status);
                    echo count($out)."<br/>";
                    for($i=0;$i<count($out);++$i)
                        echo $out[$i]."<br/>";
                    echo $status;
                }
            ?>
        </div>
    </body>
</html> 
