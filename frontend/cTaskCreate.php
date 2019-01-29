<?php
$action = $_GET["action"];
$code = $_GET["code"];

$command = "";
if ($action == "new")
{
    $command = "{\"action\":\"new\"}";
}
if ($action == "check")
{
    $command = "{\"action\":\"check\",\"code\":\"".$code."\"}";
}
if ($action == "select")
{
    $side = $_GET["side"];
    $command = "{\"action\":\"select\",\"code\":\"".$code."\",\"side\":\"".$side."\"}";
}
$id = md5(uniqid("",true));
$filename = "data/in/TASK_".$id.".txt";
$fd = fopen( $filename, "w" );
fwrite($fd, $command);
fclose( $fd );
?>
<?= $id ?>        