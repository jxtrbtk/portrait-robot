<?php
$name = $_POST["name"];
$side = $_POST["side"];
$code = $_POST["code"];
$step = $_POST["step"];

$filename = "IMG_".$side."_".$code."_".$step.".png";
if ($_FILES["data"]["name"] == $filename)
{
	$target_dir = "data/work/";
	$target_file = $target_dir . basename($_FILES["data"]["name"]);
	move_uploaded_file($_FILES["data"]["tmp_name"], $target_file);
}
else
{
	echo("name no match");
}

?>
OK