<?php header('Access-Control-Allow-Origin: *'); ?>
<?php
$id = $_GET["id"];
$filename_in = "data/in/TASK_".$id.".txt";
$filename_out = "data/out/TASK_".$id.".txt";

$status=0;
if(file_exists($filename_out)) {$status=2;}
if(file_exists($filename_in)) {$status=1;}
?>
<?= $status ?>