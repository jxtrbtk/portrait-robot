<?php 
$id = $_GET["id"];
$code = $_GET["code"];
$step = $_GET["step"];
$data = $code."_".$step;

$filename = "data/out/TASK_".$id.".txt";
$fd = fopen( $filename, "w" );
fwrite($fd, $data);
fclose( $fd );

$filename = "data/in/TASK_".$id.".txt";
unlink($filename);

?>