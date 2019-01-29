<?php 
$id = $_GET["id"];

$filename = "data/out/TASK_".$id.".txt";
unlink($filename);

?>