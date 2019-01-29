<?php 

$id = $_GET["id"];
$filename = "data/in/TASK_".$id.".txt";
$data = file_get_contents($filename);
echo $data
?>